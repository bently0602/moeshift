import Foundation

/// The bundled `moeshift` binary: the router, which starts its inference worker on demand, and the
/// `experts` tools.
enum EngineBinary {
    static func url() throws -> URL {
        if let url = Bundle.main.url(forAuxiliaryExecutable: "moeshift") { return url }
        throw AppError("The app bundle has no moeshift binary; build it with apps/build.sh.")
    }
}

/// Keeps the last lines of a process's output in memory (never on disk).
@MainActor
final class LogBuffer: ObservableObject {
    @Published private(set) var lines: [String] = []
    private var partial = ""
    private let limit = 2000

    func append(_ text: String) {
        let parts = (partial + text).components(separatedBy: "\n")
        partial = parts.last ?? ""
        lines.append(contentsOf: parts.dropLast().filter { !$0.isEmpty })
        if lines.count > limit { lines.removeFirst(lines.count - limit) }
    }

    func note(_ line: String) { append(line + "\n") }
    var text: String { (lines + (partial.isEmpty ? [] : [partial])).joined(separator: "\n") }
    var lastLine: String? { lines.last }
}

/// Starts and stops the router, and loads the model right after start so that the first real
/// request does not wait for it.
@MainActor
final class Engine: ObservableObject {
    enum State: Equatable {
        case stopped, starting, loading, ready, stopping
        case failed(String)

        var isRunning: Bool { self == .starting || self == .loading || self == .ready }
    }

    @Published private(set) var state: State = .stopped
    /// The port the running engine serves on.
    @Published private(set) var port: Int?
    let log = LogBuffer()
    private var process: Process?
    private let store: SettingsStore
    private var generation = 0
    private var afterExit: (() -> Void)?
    /// Why the engine is being stopped, when it failed: shown once it has exited.
    private var failure: String?

    init(store: SettingsStore) {
        self.store = store
        Engine.stopStale(pidURL: store.pidURL)
    }

    func start(config: URL, port: Int, alias: String, apiKey: String?) {
        guard !state.isRunning else { return }
        if Engine.portInUse(port) {
            state = .failed("Port \(port) is already in use. Choose another port.")
            return
        }
        let binary: URL
        do { binary = try EngineBinary.url() } catch {
            state = .failed(error.localizedDescription)
            return
        }
        generation += 1
        let run = generation
        failure = nil
        let process = Process()
        process.executableURL = binary
        process.arguments = ["--config", config.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.log.append(text) }
        }
        process.terminationHandler = { [weak self] finished in
            let status = finished.terminationStatus
            Task { @MainActor in self?.exited(run: run, status: status) }
        }
        log.note("— starting \(binary.lastPathComponent) on port \(port)")
        state = .starting
        do {
            try process.run()
        } catch {
            state = .failed("Cannot start the engine: \(error.localizedDescription)")
            return
        }
        self.process = process
        self.port = port
        try? "\(process.processIdentifier)".write(to: store.pidURL, atomically: true, encoding: .utf8)
        Task { await self.warmUp(run: run, port: port, alias: alias, apiKey: apiKey) }
    }

    func stop() {
        guard let process, process.isRunning else { return }
        state = .stopping
        process.terminate()
        let pid = process.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + 15) {
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        }
    }

    /// Stops the engine and calls `start` once it has exited (at once if it is not running).
    func restart(then start: @escaping () -> Void) {
        guard let process, process.isRunning else {
            start()
            return
        }
        afterExit = start
        stop()
    }

    /// For app termination: stop and wait (the worker stops itself when the router exits).
    func stopAndWait() {
        afterExit = nil
        failure = nil
        guard let process, process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(10)
        while process.isRunning && Date() < deadline { usleep(50_000) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        try? FileManager.default.removeItem(at: store.pidURL)
        generation += 1  // the exit is handled here, not by the termination handler
        self.process = nil
        port = nil
        state = .stopped
    }

    private func exited(run: Int, status: Int32) {
        guard run == generation else { return }
        process = nil
        port = nil
        try? FileManager.default.removeItem(at: store.pidURL)
        log.note("— engine exited (status \(status))")
        if let failure {
            state = .failed(failure)
            self.failure = nil
        } else if state == .stopping || status == 0 || status == SIGTERM {
            state = .stopped
        } else {
            state = .failed(log.lastLine ?? "The engine exited with status \(status).")
        }
        let next = afterExit
        afterExit = nil
        if state == .stopped { next?() }
    }

    /// Stops the engine and shows `message` once it has exited.
    private func fail(_ message: String) {
        guard let process, process.isRunning else {
            state = .failed(message)
            return
        }
        failure = message
        stop()
    }

    private func warmUp(run: Int, port: Int, alias: String, apiKey: String?) async {
        let base = "http://127.0.0.1:\(port)"
        // The router answers /health as soon as it listens.
        var healthy = false
        for _ in 0..<240 {
            if run != generation || !state.isRunning { return }
            if let (_, response) = try? await URLSession.shared.data(from: URL(string: base + "/health")!),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                healthy = true
                break
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        guard run == generation, state.isRunning else { return }
        guard healthy else {
            fail("The engine did not answer on port \(port).")
            return
        }
        state = .loading
        // One token loads the model into the worker.
        var request = URLRequest(url: URL(string: base + "/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 900
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        let body: [String: Any] = ["model": alias, "max_tokens": 1, "temperature": 0,
                                   "messages": [["role": "user", "content": "Hi"]]]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard run == generation, state == .loading else { return }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 {
                state = .ready
                log.note("— model loaded; ready")
            } else {
                fail("Loading the model failed (HTTP \(status)): \(String(decoding: data, as: UTF8.self))")
            }
        } catch {
            guard run == generation, state == .loading else { return }
            fail("Loading the model failed: \(error.localizedDescription)")
        }
    }

    static func portInUse(_ port: Int) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    /// A router left behind by an app that crashed would hold the port and the model's memory.
    static func stopStale(pidURL: URL) {
        guard let text = try? String(contentsOf: pidURL, encoding: .utf8),
              let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 else { return }
        try? FileManager.default.removeItem(at: pidURL)
        var buffer = [CChar](repeating: 0, count: 4096)
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return }
        let path = String(cString: buffer)
        if (path as NSString).lastPathComponent == "moeshift" { kill(pid, SIGTERM) }
    }
}

/// IPv4 addresses of this Mac on its local networks, for the URL other machines use.
func localNetworkAddresses() -> [String] {
    var result: [String] = []
    var list: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&list) == 0, let first = list else { return [] }
    defer { freeifaddrs(list) }
    for pointer in sequence(first: first, next: { $0.pointee.ifa_next }) {
        let entry = pointer.pointee
        guard let address = entry.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
              (Int32(entry.ifa_flags) & IFF_LOOPBACK) == 0, (Int32(entry.ifa_flags) & IFF_UP) != 0 else { continue }
        let name = String(cString: entry.ifa_name)
        guard name.hasPrefix("en") else { continue }
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        if getnameinfo(address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
            result.append(String(cString: host))
        }
    }
    return result
}

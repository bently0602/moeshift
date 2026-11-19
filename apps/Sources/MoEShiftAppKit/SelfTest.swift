import Foundation

/// Headless checks of the app's non-UI paths, run from a built bundle:
///
///   <App>.app/Contents/MacOS/<exe> --selftest-engine [--port N] [--lan] [--model-folder DIR] [--switch-port M]
///       starts the engine with the saved settings (and this port, shared on the network with the
///       saved API key, or this model folder), waits until the model is loaded, stops it. With --switch-port, it first
///       enters port M as the port field does, restarts the engine on it and checks that only M answers.
///   <App>.app/Contents/MacOS/<exe> --selftest-prepare MODEL_FOLDER DOWNLOAD_FOLDER [SOURCE_GGUF]
///       runs setup into those folders (downloading unless SOURCE_GGUF is given), keeping the download.
///
/// Neither changes the saved settings.
enum SelfTest {
    @MainActor
    static func runIfRequested(spec: ModelSpec) -> Bool {
        let args = CommandLine.arguments
        setvbuf(stdout, nil, _IOLBF, 0)  // progress lines reach a log file as they happen
        if args.contains("--selftest-engine") {
            Task { @MainActor in exit(await engine(spec: spec)) }
        } else if let i = args.firstIndex(of: "--selftest-prepare"), args.count >= i + 3 {
            let source = args.count > i + 3 ? URL(fileURLWithPath: args[i + 3]) : nil
            Task { @MainActor in exit(await prepare(spec: spec, model: args[i + 1], download: args[i + 2], source: source)) }
        } else {
            return false
        }
        dispatchMain()
    }

    @MainActor
    private static func engine(spec: ModelSpec) async -> Int32 {
        let model = AppModel(spec: spec)
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--port"), i + 1 < args.count, let port = Int(args[i + 1]) {
            model.settings.port = port
        }
        if args.contains("--lan") { model.settings.shareOnNetwork = true }
        if args.contains("--no-key") { model.settings.requireAPIKey = false }
        if let i = args.firstIndex(of: "--effort"), i + 1 < args.count { model.settings.reasoningEffort = args[i + 1] }
        if let i = args.firstIndex(of: "--model-folder"), i + 1 < args.count {
            model.settings.modelFolder = args[i + 1]
            model.refreshInstalled()
        }
        defer { model.settings = model.saved }
        guard let installed = model.installed else {
            print("selftest: no prepared model in \(model.settings.modelFolder)")
            return 1
        }
        print("selftest: model \(installed.gguf.path)")
        var start = Date()
        model.startEngine()
        guard await ready(model.engine) else { return 1 }
        print(String(format: "selftest: ready on port %d after %.1f s", model.engine.port ?? 0, Date().timeIntervalSince(start)))
        if model.settings.shareOnNetwork, let url = model.networkURLs.first.flatMap({ URL(string: $0 + "/chat/completions") }) {
            // A body that is not a JSON object is refused before inference: 401 without the key,
            // 400 once past it.
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("[]".utf8)
            let status = (try? await URLSession.shared.data(for: request)).flatMap { ($0.1 as? HTTPURLResponse)?.statusCode } ?? 0
            let expected = model.settings.requireAPIKey ? 401 : 400
            print("selftest: \(url.absoluteString) without a key: HTTP \(status), expected \(expected)")
            guard status == expected else {
                model.engine.stopAndWait()
                return 1
            }
        }
        if let i = args.firstIndex(of: "--switch-port"), i + 1 < args.count {
            let old = model.engine.port ?? 0
            model.portText = args[i + 1]
            model.applyPort()
            if let problem = model.portProblem {
                print("selftest: failed: \(problem)")
                model.engine.stopAndWait()
                return 1
            }
            start = Date()
            model.restartEngine()
            while model.engine.state == .ready { try? await Task.sleep(nanoseconds: 50_000_000) }
            guard await ready(model.engine) else { return 1 }
            let port = model.engine.port ?? 0
            print(String(format: "selftest: ready on port %d after %.1f s; old port %d answers: %@, new port answers: %@",
                         port, Date().timeIntervalSince(start), old, String(Engine.portInUse(old)), String(Engine.portInUse(port))))
            guard port == model.settings.port, !Engine.portInUse(old), Engine.portInUse(port) else {
                model.engine.stopAndWait()
                return 1
            }
        }
        model.engine.stopAndWait()
        print("selftest: stopped; running = \(model.engine.state.isRunning)")
        return 0
    }

    @MainActor
    private static func ready(_ engine: Engine) async -> Bool {
        while true {
            switch engine.state {
            case .ready:
                return true
            case .failed(let message):
                print("selftest: failed: \(message)\n\(engine.log.text)")
                return false
            default:
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
    }

    @MainActor
    private static func prepare(spec: ModelSpec, model folder: String, download: String, source: URL?) async -> Int32 {
        var settings = AppSettings.defaults(for: spec)
        settings.modelFolder = folder
        settings.downloadFolder = download
        settings.deleteDownload = false
        let preparer = Preparer(spec: spec)
        var lastStep = ""
        var lastReport = Date.distantPast
        var result: Int32?
        preparer.start(settings: settings, source: source) { installed in
            print("selftest: prepared \(installed.gguf.path)")
            result = 0
        }
        while result == nil {
            if case .failed(let message) = preparer.phase {
                print("selftest: failed: \(message)")
                result = 1
                break
            }
            if preparer.step != lastStep || Date().timeIntervalSince(lastReport) > 30 {
                lastStep = preparer.step
                lastReport = Date()
                print("selftest: \(preparer.step) \(preparer.detail)")
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        return result ?? 1
    }
}

import Foundation

/// First-run setup: download the GGUF (or use one the user has), then `moeshift experts extract`,
/// `verify` and `strip` into the model folder, then optionally delete the download.
@MainActor
final class Preparer: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running
        case failed(String)
        case done
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var step = ""
    @Published private(set) var detail = ""
    @Published private(set) var fraction: Double?
    let log = LogBuffer()

    private let spec: ModelSpec
    private var task: Task<Void, Never>?
    private var tool: Process?

    init(spec: ModelSpec) { self.spec = spec }

    /// `source`: the first shard of a GGUF the user already has; nil downloads into the download folder.
    func start(settings: AppSettings, source: URL?, finished: @escaping (InstalledModel) -> Void) {
        guard phase != .running else { return }
        phase = .running
        task = Task {
            do {
                let model = try await prepare(settings: settings, source: source)
                phase = .done
                finished(model)
            } catch is CancellationError {
                phase = .idle
                step = ""
                detail = ""
            } catch {
                phase = .failed(error.localizedDescription)
                log.note("— failed: \(error.localizedDescription)")
            }
            fraction = nil
        }
    }

    func cancel() {
        task?.cancel()
        tool?.terminate()
    }

    private func prepare(settings: AppSettings, source: URL?) async throws -> InstalledModel {
        let modelFolder = URL(fileURLWithPath: settings.modelFolder)
        try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)

        // 1. The GGUF: the user's, one already on disk, or a download.
        var downloaded: [URL] = []
        let first: URL
        var source = source
        if source == nil, let found = OriginalGGUF.find(spec: spec, modelFolder: settings.modelFolder) {
            log.note("— using the GGUF already in \(found.deletingLastPathComponent().path); no download")
            source = found
        }
        if let source {
            first = source
            try checkShards(next: source)
        } else {
            let folder = URL(fileURLWithPath: settings.downloadFolder)
            try checkSpace(settings: settings)
            for (index, file) in spec.files.enumerated() {
                let destination = folder.appendingPathComponent(file.name)
                downloaded.append(destination)
                if size(of: destination) == file.size { continue }
                step = spec.files.count > 1 ? "Downloading part \(index + 1) of \(spec.files.count)" : "Downloading"
                log.note("— downloading \(file.name) to \(folder.path)")
                let downloader = Downloader(url: spec.url(for: file), destination: destination, size: file.size,
                                            sha256: file.sha256) { progress in
                    Task { @MainActor in self.show(progress) }
                }
                try await downloader.run()
                try Task.checkCancellation()
            }
            first = downloaded[0]
        }

        // 2. Experts: extract unless a complete, verified folder is already there.
        let experts = modelFolder.appendingPathComponent("experts")
        let manifest = experts.appendingPathComponent("manifest.json")
        var verified = false
        if FileManager.default.fileExists(atPath: manifest.path) {
            step = "Checking existing experts"
            verified = (try? await runTool(["verify", "--model", first.path, "--experts", experts.path],
                                           progress: "verified tensor")) != nil
        }
        if !verified {
            step = "Extracting experts"
            try await runTool(["extract", "--model", first.path, "--out-dir", experts.path, "--force"],
                              progress: "extracted layer")
            step = "Verifying experts"
            try await runTool(["verify", "--model", first.path, "--experts", experts.path],
                              progress: "verified tensor")
        }

        // 3. The GGUF without experts (checked against the original as it is written).
        let stripped = modelFolder.appendingPathComponent(spec.strippedName)
        if !FileManager.default.fileExists(atPath: stripped.path) {
            step = "Writing the model file without experts"
            try await runTool(["strip", "--model", first.path, "--experts", experts.path, "--out", stripped.path],
                              progress: nil)
        }

        // 4. Only files this app downloaded are ever deleted.
        if settings.deleteDownload {
            for url in downloaded {
                try? FileManager.default.removeItem(at: url)
                log.note("— deleted \(url.path)")
            }
        }
        step = "Done"
        detail = ""
        return InstalledModel(gguf: stripped, experts: experts, stripped: true)
    }

    private func show(_ progress: Downloader.Progress) {
        fraction = progress.total > 0 ? Double(progress.done) / Double(progress.total) : nil
        if progress.bytesPerSecond > 0 {
            let left = Double(progress.total - progress.done) / progress.bytesPerSecond
            detail = String(format: "%@ of %@ · %.0f MB/s · %@ left", formatBytes(progress.done), formatBytes(progress.total),
                            progress.bytesPerSecond / 1e6, formatDuration(left))
        } else {
            detail = "Checking the \(formatBytes(progress.done)) already downloaded"
        }
    }

    private func checkSpace(settings: AppSettings) throws {
        let remaining = spec.files.reduce(Int64(0)) { sum, file in
            let url = URL(fileURLWithPath: settings.downloadFolder).appendingPathComponent(file.name)
            let have = size(of: url) == file.size ? file.size : size(of: url.appendingPathExtension("part"))
            return sum + file.size - have
        }
        let kept = spec.keptBytes
        if Volumes.sameVolume(settings.downloadFolder, settings.modelFolder) {
            if let free = Volumes.freeBytes(at: settings.modelFolder), free < remaining + kept {
                throw AppError("This drive needs \(formatBytes(remaining + kept)) free for the download and the model files; \(formatBytes(free)) is free.")
            }
        } else {
            if let free = Volumes.freeBytes(at: settings.downloadFolder), free < remaining {
                throw AppError("The download folder's drive needs \(formatBytes(remaining)) free; \(formatBytes(free)) is free.")
            }
            if let free = Volumes.freeBytes(at: settings.modelFolder), free < kept {
                throw AppError("The model folder's drive needs \(formatBytes(kept)) free; \(formatBytes(free)) is free.")
            }
        }
    }

    /// A user-chosen GGUF must be this model's first shard, with every shard beside it.
    private func checkShards(next first: URL) throws {
        guard first.lastPathComponent == spec.files[0].name else {
            throw AppError("Choose \(spec.files[0].name).")
        }
        for file in spec.files {
            let url = first.deletingLastPathComponent().appendingPathComponent(file.name)
            guard size(of: url) == file.size else {
                throw AppError("\(file.name) is missing or incomplete next to the chosen file.")
            }
        }
    }

    /// Runs `moeshift experts <args>`; lines starting with "moeshift: <progress> i/n" move the bar.
    @discardableResult
    private func runTool(_ args: [String], progress: String?) async throws -> Int32 {
        let process = Process()
        process.executableURL = try EngineBinary.url()
        process.arguments = ["experts"] + args
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        fraction = progress == nil ? nil : 0
        detail = ""
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                guard let self else { return }
                self.log.append(text)
                if let progress, let line = text.split(separator: "\n").last(where: { $0.contains(progress) }),
                   let counts = line.split(separator: " ").last?.split(separator: "/"), counts.count == 2,
                   let done = Double(counts[0]), let total = Double(counts[1]), total > 0 {
                    self.fraction = done / total
                    self.detail = "\(Int(done)) of \(Int(total))"
                }
            }
        }
        log.note("— moeshift experts \(args.joined(separator: " "))")
        let status: Int32 = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                process.terminationHandler = { finished in continuation.resume(returning: finished.terminationStatus) }
                do { try process.run() } catch { continuation.resume(throwing: error) }
                Task { @MainActor in self.tool = process }
            }
        } onCancel: {
            process.terminate()
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        tool = nil
        try Task.checkCancellation()
        guard status == 0 else {
            throw AppError(log.lastLine?.replacingOccurrences(of: "moeshift: ", with: "") ?? "moeshift experts \(args[0]) failed.")
        }
        return status
    }

    private func size(of url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
    }
}

func formatDuration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds > 0 else { return "—" }
    let s = Int(seconds)
    return s >= 3600 ? "\(s / 3600) h \(s % 3600 / 60) min" : s >= 60 ? "\(s / 60) min" : "\(s) s"
}

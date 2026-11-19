import Foundation
import Security

/// What the user chose. Saved as settings.json (mode 0600, since it holds the API key) in
/// ~/Library/Application Support/MoEShift/<alias>/.
public struct AppSettings: Codable, Equatable {
    public var modelFolder: String
    public var downloadFolder: String
    public var port: Int
    public var shareOnNetwork: Bool
    /// Whether network clients must send the API key. Only used while sharing on the network.
    public var requireAPIKey: Bool
    public var apiKey: String
    public var deleteDownload: Bool
    /// Reasoning effort for requests that set none: none, low, medium or high (thinking unlimited).
    /// nil until the user picks one: the model's default.
    public var reasoningEffort: String?

    static func defaults(for spec: ModelSpec) -> AppSettings {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let models = home.appendingPathComponent(".moeshift/models/\(spec.alias)")
        return AppSettings(
            modelFolder: models.path,
            downloadFolder: Volumes.defaultDownloadFolder(needing: spec.downloadBytes).path,
            port: spec.defaultPort,
            shareOnNetwork: false,
            requireAPIKey: true,
            apiKey: newAPIKey(),
            deleteDownload: true,
            reasoningEffort: nil)
    }

    /// The key the engine requires, or nil when it requires none.
    var engineAPIKey: String? { shareOnNetwork && requireAPIKey ? apiKey : nil }
}

extension AppSettings {
    /// Settings saved before requireAPIKey existed still load, with the key required.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelFolder = try c.decode(String.self, forKey: .modelFolder)
        downloadFolder = try c.decode(String.self, forKey: .downloadFolder)
        port = try c.decode(Int.self, forKey: .port)
        shareOnNetwork = try c.decode(Bool.self, forKey: .shareOnNetwork)
        requireAPIKey = try c.decodeIfPresent(Bool.self, forKey: .requireAPIKey) ?? true
        apiKey = try c.decode(String.self, forKey: .apiKey)
        deleteDownload = try c.decode(Bool.self, forKey: .deleteDownload)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
    }
}

/// 48 hex characters from the system's secure random generator.
func newAPIKey() -> String {
    var bytes = [UInt8](repeating: 0, count: 24)
    _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
    return bytes.map { String(format: "%02x", $0) }.joined()
}

struct SettingsStore {
    let folder: URL

    init(spec: ModelSpec) {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        folder = support.appendingPathComponent("MoEShift/\(spec.alias)")
    }

    var settingsURL: URL { folder.appendingPathComponent("settings.json") }
    var configURL: URL { folder.appendingPathComponent("config.json") }
    var pidURL: URL { folder.appendingPathComponent("engine.pid") }

    func load() -> AppSettings? {
        guard let data = try? Data(contentsOf: settingsURL) else { return nil }
        return try? JSONDecoder().decode(AppSettings.self, from: data)
    }

    func save(_ settings: AppSettings) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try writePrivate(try encoder.encode(settings), to: settingsURL)
    }

    func writePrivate(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

/// Free space and the default download folder.
enum Volumes {
    static func freeBytes(at path: String) -> Int64? {
        var url = URL(fileURLWithPath: path)
        // The folder may not exist yet: ask the nearest existing parent.
        while !FileManager.default.fileExists(atPath: url.path) && url.path != "/" {
            url.deleteLastPathComponent()
        }
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    static func sameVolume(_ a: String, _ b: String) -> Bool {
        func volume(_ path: String) -> URL? {
            var url = URL(fileURLWithPath: path)
            while !FileManager.default.fileExists(atPath: url.path) && url.path != "/" {
                url.deleteLastPathComponent()
            }
            return (try? url.resourceValues(forKeys: [.volumeURLKey]))?.volume
        }
        return volume(a) != nil && volume(a) == volume(b)
    }

    /// The external local drive with the most free space, if it can hold the download; otherwise
    /// ~/Downloads. Downloading to another drive spares the internal SSD a full extra write.
    static func defaultDownloadFolder(needing bytes: Int64) -> URL {
        let keys: [URLResourceKey] = [.volumeIsInternalKey, .volumeIsLocalKey, .volumeIsReadOnlyKey,
                                      .volumeIsRootFileSystemKey, .volumeAvailableCapacityForImportantUsageKey]
        let volumes = FileManager.default.mountedVolumeURLs(includingResourceValuesForKeys: keys,
                                                           options: [.skipHiddenVolumes]) ?? []
        let best = volumes.compactMap { url -> (URL, Int64)? in
            guard let v = try? url.resourceValues(forKeys: Set(keys)),
                  v.volumeIsLocal == true, v.volumeIsReadOnly != true, v.volumeIsRootFileSystem != true,
                  v.volumeIsInternal != true,
                  let free = v.volumeAvailableCapacityForImportantUsage, free > bytes + (10 << 30) else { return nil }
            return (url, free)
        }.max { $0.1 < $1.1 }
        if let best { return best.0.appendingPathComponent("MoEShift Downloads") }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Downloads/MoEShift")
    }
}

/// The prepared files in a model folder: experts/ with its manifest, and the stripped GGUF or, from
/// an earlier setup, the original one (all shards).
struct InstalledModel: Equatable {
    let gguf: URL
    let experts: URL
    let stripped: Bool

    static func find(spec: ModelSpec, in folder: String) -> InstalledModel? {
        let base = URL(fileURLWithPath: folder)
        let experts = base.appendingPathComponent("experts")
        let fm = FileManager.default
        guard fm.fileExists(atPath: experts.appendingPathComponent("manifest.json").path) else { return nil }
        let stripped = base.appendingPathComponent(spec.strippedName)
        if fm.fileExists(atPath: stripped.path) {
            return InstalledModel(gguf: stripped, experts: experts, stripped: true)
        }
        let shards = spec.files.map { base.appendingPathComponent($0.name) }
        let complete = zip(shards, spec.files).allSatisfy { url, file in
            ((try? fm.attributesOfItem(atPath: url.path))?[.size] as? Int64) == file.size
        }
        return complete ? InstalledModel(gguf: shards[0], experts: experts, stripped: false) : nil
    }
}

/// A complete original GGUF (every shard at its exact size) that is already on disk, so setup can
/// use it instead of downloading: in the model folder, or where the experts were extracted from
/// (manifest.json records the source path).
enum OriginalGGUF {
    static func find(spec: ModelSpec, modelFolder: String) -> URL? {
        var folders = [URL(fileURLWithPath: modelFolder)]
        let manifest = URL(fileURLWithPath: modelFolder).appendingPathComponent("experts/manifest.json")
        if let data = try? Data(contentsOf: manifest),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let source = json["source"] as? [String: Any],
           let files = source["model_files"] as? [String], let first = files.first {
            folders.append(URL(fileURLWithPath: first).deletingLastPathComponent())
        }
        let fm = FileManager.default
        for folder in folders {
            let complete = spec.files.allSatisfy { file in
                ((try? fm.attributesOfItem(atPath: folder.appendingPathComponent(file.name).path))?[.size] as? Int64) == file.size
            }
            if complete { return folder.appendingPathComponent(spec.files[0].name) }
        }
        return nil
    }
}

/// Writes the router configuration from the bundled profile and the user's choices.
enum ConfigWriter {
    static func profileTemplate() throws -> URL {
        if let url = Bundle.main.url(forResource: "moeshift", withExtension: "json") { return url }
        throw AppError("The app bundle has no moeshift.json; build it with apps/build.sh.")
    }

    static func write(spec: ModelSpec, model: InstalledModel, settings: AppSettings, store: SettingsStore) throws {
        let data = try Data(contentsOf: try profileTemplate())
        guard let template = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = template["models"] as? [String: Any],
              var profile = models[spec.alias] as? [String: Any],
              var args = profile["args"] as? [String: Any] else {
            throw AppError("moeshift.json has no profile for \(spec.alias).")
        }
        args["model"] = model.gguf.path
        args["moe-experts"] = model.experts.path
        args["reasoning-effort"] = settings.reasoningEffort ?? spec.defaultReasoningEffort
        profile["args"] = args
        var config: [String: Any] = [
            "host": settings.shareOnNetwork ? "0.0.0.0" : "127.0.0.1",
            "port": settings.port,
            "models": [spec.alias: profile],
        ]
        if let key = settings.engineAPIKey { config["api-key"] = key }
        else if settings.shareOnNetwork { config["no-api-key"] = true }
        let out = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try store.writePrivate(out, to: store.configURL)
    }
}

struct AppError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

func formatBytes(_ bytes: Int64) -> String {
    String(format: "%.1f GB", Double(bytes) / 1e9)
}

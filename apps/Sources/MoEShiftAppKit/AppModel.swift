import AppKit
import Foundation
import SwiftUI

/// The state behind one app's window.
@MainActor
final class AppModel: ObservableObject {
    let spec: ModelSpec
    let store: SettingsStore
    let engine: Engine
    let preparer: Preparer

    @Published var settings: AppSettings {
        didSet { if settings != oldValue { try? store.save(settings) } }
    }
    /// The settings as loaded, for the self-test to restore.
    private(set) var saved: AppSettings
    @Published private(set) var installed: InstalledModel?
    @Published var showingSetup = false
    // View state lives here: SwiftUI's @State is a macro, and the Command Line Tools ship no macro plugins.
    @Published var portText = ""
    @Published private(set) var portProblem: String?
    @Published var showLog = false

    init(spec: ModelSpec) {
        self.spec = spec
        store = SettingsStore(spec: spec)
        engine = Engine(store: store)
        preparer = Preparer(spec: spec)
        let loaded = store.load()
        let initial = loaded ?? AppSettings.defaults(for: spec)
        settings = initial
        saved = initial
        if loaded == nil { try? store.save(settings) }
        portText = String(settings.port)
        refreshInstalled()
    }

    func refreshInstalled() {
        installed = InstalledModel.find(spec: spec, in: settings.modelFolder)
    }

    /// Saves the port field when it holds a usable port, and says what is wrong with it otherwise.
    func applyPort() {
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)), (1024...65535).contains(port) else {
            portProblem = "Enter a port from 1024 to 65535."
            return
        }
        settings.port = port
        portProblem = port != engine.port && Engine.portInUse(port) ? "Port \(port) is in use by another program." : nil
    }

    /// Stops the engine and starts it again with the current settings (a new port).
    func restartEngine() {
        engine.restart { [weak self] in self?.startEngine() }
    }

    func startEngine() {
        if Int(portText) != settings.port {
            portText = String(settings.port)
            portProblem = nil
        }
        refreshInstalled()
        guard let installed else {
            showingSetup = true
            return
        }
        do {
            try ConfigWriter.write(spec: spec, model: installed, settings: settings, store: store)
        } catch {
            engine.log.note("— \(error.localizedDescription)")
            return
        }
        engine.start(config: store.configURL, port: settings.port, alias: spec.alias,
                     apiKey: settings.engineAPIKey)
    }

    func prepare(source: URL?) {
        preparer.start(settings: settings, source: source) { [weak self] _ in
            self?.refreshInstalled()
            self?.showingSetup = false
        }
    }

    /// The engine's port while it runs, otherwise the one it will start on.
    var activePort: Int { engine.port ?? settings.port }
    var localURL: String { "http://127.0.0.1:\(activePort)/v1" }
    var networkURLs: [String] { localNetworkAddresses().map { "http://\($0):\(activePort)/v1" } }
}

func copyToPasteboard(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}

/// The app entry point both app targets call with their model.
public enum MoEShiftLauncher {
    @MainActor static var model: AppModel!

    @MainActor
    public static func main(spec: ModelSpec) {
        if SelfTest.runIfRequested(spec: spec) { return }
        model = AppModel(spec: spec)
        MoEShiftApp.main()
    }
}

final class MoEShiftAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            MoEShiftLauncher.model?.preparer.cancel()
            MoEShiftLauncher.model?.engine.stopAndWait()
        }
    }
}

struct MoEShiftApp: App {
    @NSApplicationDelegateAdaptor(MoEShiftAppDelegate.self) private var delegate

    var body: some Scene {
        Window(MoEShiftLauncher.model.spec.displayName, id: "main") {
            RootView(model: MoEShiftLauncher.model)
        }
        .windowResizability(.contentSize)
    }
}

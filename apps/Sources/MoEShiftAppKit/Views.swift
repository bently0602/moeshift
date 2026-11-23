import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Group {
            if model.installed == nil || model.showingSetup {
                SetupView(model: model, preparer: model.preparer)
            } else {
                MainView(model: model, engine: model.engine)
            }
        }
        .frame(width: 560)
    }
}

// MARK: - Main window

struct MainView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var engine: Engine

    var body: some View {
        Form {
            Section {
                HStack(spacing: 10) {
                    Circle().fill(statusColor).frame(width: 12, height: 12)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(statusTitle).font(.headline)
                        if case .failed(let message) = engine.state {
                            Text(message).font(.caption).foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled)
                        }
                    }
                    Spacer()
                    if engine.state.isRunning || engine.state == .stopping {
                        Button("Stop") { engine.stop() }.disabled(engine.state == .stopping)
                    } else {
                        Button("Start") { model.startEngine() }.keyboardShortcut(.defaultAction)
                    }
                }
            }

            Section("Server") {
                LabeledContent("Port") {
                    HStack {
                        if portChanged, model.portProblem == nil {
                            Button("Restart on port \(String(model.settings.port))") { model.restartEngine() }
                                .disabled(engine.state == .stopping)
                        }
                        TextField("", text: $model.portText)
                            .frame(width: 80)
                            .multilineTextAlignment(.trailing)
                            .onChange(of: model.portText) { model.applyPort() }
                    }
                }
                if let problem = model.portProblem {
                    Text(problem).font(.caption).foregroundStyle(.red)
                } else if portChanged, let running = engine.port {
                    Text("The engine is serving on port \(String(running)) until it restarts.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Share on the local network", isOn: $model.settings.shareOnNetwork)
                    .disabled(locked)
                if model.settings.shareOnNetwork {
                    Toggle("Require the API key", isOn: $model.settings.requireAPIKey)
                        .disabled(locked)
                    if model.settings.requireAPIKey {
                        Text("Other computers on your network can use the model with the API key below. macOS may ask whether moeshift may accept incoming connections.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Anyone on your network can use the model, without a key. macOS may ask whether moeshift may accept incoming connections.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
            }

            Section("Connect a client") {
                CopyRow(label: "OpenAI URL", value: model.localURL)
                CopyRow(label: "Anthropic URL", value: model.localBaseURL)
                if model.settings.shareOnNetwork {
                    ForEach(model.networkBaseURLs, id: \.self) { base in
                        CopyRow(label: "Network, OpenAI", value: base + "/v1")
                        CopyRow(label: "Network, Anthropic", value: base)
                    }
                }
                Text(model.settings.engineAPIKey != nil
                     ? "OpenAI clients (chat completions, Responses) use the OpenAI URL; Anthropic clients (Messages) use the Anthropic URL. Send the key as an API key: Authorization: Bearer or x-api-key."
                     : "OpenAI clients (chat completions, Responses) use the OpenAI URL; Anthropic clients (Messages) use the Anthropic URL. Any API key a client insists on is accepted.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.settings.engineAPIKey != nil {
                    CopyRow(label: "API key", value: model.settings.apiKey, monospaced: true) {
                        Button("New key") { model.settings.apiKey = newAPIKey() }.disabled(locked)
                    }
                }
                CopyRow(label: "Model", value: model.spec.alias, monospaced: true)
            }

            Section("Model") {
                LabeledContent("Files") {
                    HStack {
                        Text(model.settings.modelFolder).lineLimit(1).truncationMode(.middle)
                        Button("Show") { NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: model.settings.modelFolder) }
                    }
                }
                Text("\(model.spec.displayName) (\(model.spec.tagline)), with settings measured on a Mac Studio M5 Max with 36 GB. One request runs at a time.")
                    .font(.caption).foregroundStyle(.secondary)
                Picker("Thinking", selection: Binding(
                    get: { model.settings.reasoningEffort ?? model.spec.defaultReasoningEffort },
                    set: { model.settings.reasoningEffort = $0 })) {
                    Text("Off").tag("none")
                    Text("Low (\(budgetLabel(model.settings.lowBudget ?? AppSettings.defaultLowBudget)))").tag("low")
                    Text("Medium (\(budgetLabel(model.settings.mediumBudget ?? AppSettings.defaultMediumBudget)))").tag("medium")
                    Text("High (\(budgetLabel(model.settings.highBudget)))").tag("high")
                }
                .disabled(locked)
                Text("How long the model may think before answering. A client that sets its own reasoning effort or thinking budget overrides this.")
                    .font(.caption).foregroundStyle(.secondary)
                DisclosureGroup("Thinking limits") {
                    ThinkingLimits(settings: $model.settings).disabled(locked)
                }
                Picker("Context", selection: Binding(
                    get: { model.settings.contextSize ?? AppSettings.contextSizes.last! },
                    set: { model.settings.contextSize = $0 })) {
                    ForEach(AppSettings.contextSizes, id: \.self) { size in
                        Text("\(size / 1024)k tokens").tag(size)
                    }
                }
                .disabled(locked)
                Text("The longest conversation the engine accepts: prompt, thinking and answer together. Longer requests are refused, and an answer that reaches the limit is cut off. Memory grows with the conversation, so a lower limit caps it.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Change model files…") { model.showingSetup = true }.disabled(locked)
            }

            Section {
                DisclosureGroup("Log", isExpanded: $model.showLog) {
                    LogView(log: engine.log)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var locked: Bool { engine.state.isRunning || engine.state == .stopping }
    /// The port was changed while the engine runs on another one.
    private var portChanged: Bool { engine.port.map { $0 != model.settings.port } ?? false }

    private var statusTitle: String {
        switch engine.state {
        case .stopped: return "Stopped"
        case .starting: return "Starting…"
        case .loading: return "Loading the model…"
        case .ready: return "Ready"
        case .stopping: return "Stopping…"
        case .failed: return "Stopped with an error"
        }
    }

    private var statusColor: Color {
        switch engine.state {
        case .ready: return .green
        case .starting, .loading, .stopping: return .orange
        case .failed: return .red
        case .stopped: return .gray
        }
    }
}

/// "up to N tokens", or "no limit" for nil.
func budgetLabel(_ tokens: Int?) -> String {
    guard let tokens else { return "no limit" }
    return "up to \(tokens.formatted()) tokens"
}

/// The thinking tokens that Low, Medium and High stand for, for this app's Thinking setting and for
/// clients that send that reasoning effort.
struct ThinkingLimits: View {
    @Binding var settings: AppSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            BudgetField(label: "Low", value: binding(\.lowBudget, default: AppSettings.defaultLowBudget))
            BudgetField(label: "Medium", value: binding(\.mediumBudget, default: AppSettings.defaultMediumBudget))
            LabeledContent("High") {
                HStack {
                    Toggle("Limit", isOn: Binding(
                        get: { settings.highBudget != nil },
                        set: { settings.highBudget = $0 ? max(settings.mediumBudget ?? AppSettings.defaultMediumBudget, 16_384) : nil }))
                    if settings.highBudget != nil {
                        TokenField(value: Binding(get: { settings.highBudget ?? 16_384 }, set: { settings.highBudget = $0 }))
                    } else {
                        Text("no limit").foregroundStyle(.secondary)
                    }
                }
            }
            if let warning {
                Text(warning).font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Text("Tokens of thinking, from \(AppSettings.budgetRange.lowerBound.formatted()) to \(AppSettings.budgetRange.upperBound.formatted()). Extra high and Max stay unlimited.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Restore defaults") {
                    settings.lowBudget = nil
                    settings.mediumBudget = nil
                    settings.highBudget = nil
                }
                .disabled(settings.lowBudget == nil && settings.mediumBudget == nil && settings.highBudget == nil)
            }
        }
        .padding(.top, 4)
    }

    /// A budget equal to the engine's default is stored as nil, so the configuration stays default.
    private func binding(_ key: WritableKeyPath<AppSettings, Int?>, default value: Int) -> Binding<Int> {
        Binding(get: { settings[keyPath: key] ?? value },
                set: { settings[keyPath: key] = $0 == value ? nil : $0 })
    }

    private var warning: String? {
        let low = settings.lowBudget ?? AppSettings.defaultLowBudget
        let medium = settings.mediumBudget ?? AppSettings.defaultMediumBudget
        if low > medium { return "Low allows more thinking than Medium." }
        if let high = settings.highBudget, medium > high { return "Medium allows more thinking than High." }
        return nil
    }
}

struct BudgetField: View {
    let label: String
    @Binding var value: Int

    var body: some View {
        LabeledContent(label) { TokenField(value: $value) }
    }
}

/// A token count, kept within AppSettings.budgetRange.
struct TokenField: View {
    @Binding var value: Int

    var body: some View {
        HStack(spacing: 4) {
            TextField("", value: Binding(
                get: { value },
                set: { value = min(max($0, AppSettings.budgetRange.lowerBound), AppSettings.budgetRange.upperBound) }),
                format: .number)
                .frame(width: 80)
                .multilineTextAlignment(.trailing)
            Text("tokens").foregroundStyle(.secondary)
        }
    }
}

struct CopyRow<Extra: View>: View {
    let label: String
    let value: String
    var monospaced = false
    @ViewBuilder var extra: () -> Extra

    var body: some View {
        LabeledContent(label) {
            HStack {
                Text(value)
                    .font(monospaced ? .system(.body, design: .monospaced) : .body)
                    .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                extra()
                Button { copyToPasteboard(value) } label: { Image(systemName: "doc.on.doc") }
                    .buttonStyle(.borderless).help("Copy")
            }
        }
    }
}

extension CopyRow where Extra == EmptyView {
    init(label: String, value: String, monospaced: Bool = false) {
        self.init(label: label, value: value, monospaced: monospaced) { EmptyView() }
    }
}

struct LogView: View {
    @ObservedObject var log: LogBuffer

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(log.text.isEmpty ? "No output yet." : log.text)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .id("end")
            }
            .frame(height: 180)
            .onChange(of: log.lines.count) { proxy.scrollTo("end", anchor: .bottom) }
        }
    }
}

// MARK: - Setup

struct SetupView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var preparer: Preparer

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Set up \(model.spec.displayName)").font(.title2.bold())
                    Text("The app downloads \(formatBytes(model.spec.downloadBytes)) from Hugging Face (\(model.spec.repo)), checks it, and prepares \(formatBytes(model.spec.keptBytes)) of model files: the experts the engine reads from the SSD for each token, and the rest of the model in one smaller file. The download is not needed afterwards.")
                        .foregroundStyle(.secondary)
                }
            }

            if let found = InstalledModel.find(spec: model.spec, in: model.settings.modelFolder), preparer.phase != .running {
                Section {
                    HStack {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                        Text(found.stripped ? "Prepared model files are already in this folder." : "Prepared experts and the original GGUF are already in this folder.")
                        Spacer()
                        Button("Use them") { model.refreshInstalled(); model.showingSetup = false }
                    }
                }
            }

            Section("Model files (kept)") {
                FolderRow(path: $model.settings.modelFolder, needed: model.spec.keptBytes,
                          note: "Put these on the internal SSD: the engine reads experts from it for every token.")
                    .disabled(busy)
            }

            Section("Download (temporary)") {
                FolderRow(path: $model.settings.downloadFolder, needed: model.spec.downloadBytes,
                          note: "Another drive keeps the download from being written to the internal SSD. A stopped download resumes.")
                    .disabled(busy)
                Toggle("Delete the download after the model files are prepared and checked", isOn: $model.settings.deleteDownload)
                    .disabled(busy)
            }

            Section {
                switch preparer.phase {
                case .running:
                    VStack(alignment: .leading, spacing: 6) {
                        Text(preparer.step).font(.headline)
                        if let fraction = preparer.fraction {
                            ProgressView(value: fraction)
                        } else {
                            ProgressView().progressViewStyle(.linear)
                        }
                        Text(preparer.detail).font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Spacer()
                            Button("Cancel") { preparer.cancel() }
                        }
                    }
                case .failed(let message):
                    Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red).textSelection(.enabled)
                    buttons
                default:
                    buttons
                }
            }

            Section {
                DisclosureGroup("Details") { LogView(log: preparer.log) }
            }
        }
        .formStyle(.grouped)
    }

    private var busy: Bool { preparer.phase == .running }

    @ViewBuilder private var buttons: some View {
        HStack {
            if model.installed != nil {
                Button("Back") { model.showingSetup = false }
            }
            Button("Use a GGUF I already downloaded…", action: chooseGGUF)
            Spacer()
            if let existing = OriginalGGUF.find(spec: model.spec, modelFolder: model.settings.modelFolder) {
                Button("Prepare from the GGUF on disk") { model.prepare(source: existing) }
                    .keyboardShortcut(.defaultAction)
                    .help(existing.deletingLastPathComponent().path)
            } else {
                Button("Download and Prepare") { model.prepare(source: nil) }.keyboardShortcut(.defaultAction)
            }
        }
    }

    private func chooseGGUF() {
        let panel = NSOpenPanel()
        panel.message = "Choose \(model.spec.files[0].name)"
        panel.allowedContentTypes = []
        panel.allowsOtherFileTypes = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            model.prepare(source: url)
        }
    }
}

struct FolderRow: View {
    @Binding var path: String
    let needed: Int64
    let note: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(path).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                Spacer()
                Button("Choose…", action: choose)
            }
            Text(spaceText + " " + note).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var spaceText: String {
        guard let free = Volumes.freeBytes(at: path) else { return "Needs \(formatBytes(needed))." }
        return "Needs \(formatBytes(needed)); \(formatBytes(free)) free."
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: path).deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url { path = url.path }
    }
}

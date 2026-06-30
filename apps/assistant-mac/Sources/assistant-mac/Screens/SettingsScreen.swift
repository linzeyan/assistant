import AppKit
import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject var controller: BackendController
    // Absorbed from the former Setup tab: managed-tool install + first-run bootstrap now
    // live here so there's one place for "configure the backend" (N6).
    @StateObject private var boot = BackendBootstrap()
    @State private var preflight: PreflightDTO?
    @State private var draft: String = ""
    @State private var modelsDir: String = ""
    @State private var downloadDir: String = ""
    @State private var extraModelDirs: [String] = []
    @State private var hfCache: Bool = false
    @State private var backendHost: String = ""
    @State private var backendPort: String = ""
    @State private var modelBackend: String = "mlx"
    @State private var maxOutputTokens: String = ""
    @State private var maxToolIters: String = ""
    @State private var turnTimeoutS: String = ""
    @State private var memCeilingGb: String = ""
    @State private var configPath: String = ""
    @State private var savedNote: String?
    @State private var bindNote: String?
    @State private var backendKindNote: String?
    @State private var agentNote: String?

    @State private var fusionEnabled = false
    @State private var fusionPanel: Set<String> = []
    @State private var fusionJudge: String?
    // Gateways (S9): the token field is write-only (blank = keep current); the rest is status.
    @State private var telegramTokenInput: String = ""
    @State private var telegramAllowlist: String = ""
    @State private var telegramConfigured: Bool = false
    @State private var telegramTokenMasked: String?
    @State private var telegramRunning: Bool = false
    @State private var telegramError: String?
    @State private var gatewaysNote: String?
    // Resolved (read-only) runtime paths, so the user can see what's actually in use.
    @State private var backendExe: String = ""
    @State private var inUseVenv: String = ""
    @State private var resolvedUv: String = ""
    @State private var resolvedVenv: String = ""

    // Managed-venv bootstrap overrides (read by `Bootstrap`). Blank = sensible default.
    @AppStorage("uvPath") private var uvPath: String = ""
    @AppStorage("venvPath") private var venvPath: String = ""
    @AppStorage("pythonVersion") private var pythonVersion: String = "3.12"

    private let labelWidth: CGFloat = 150

    var body: some View {
        // Split into tabs (B1): the single Form had grown to ~11 sections. Grouping by
        // concern — Backend / Agent / Gateways / Advanced — keeps each screen scannable.
        // load() + the preflight poll live on the TabView so every tab shares one refresh.
        TabView {
            backendTab.tabItem { Label("Backend", systemImage: "server.rack") }
            agentTab.tabItem { Label("Agent", systemImage: "brain") }
            gatewaysTab.tabItem {
                Label("Gateways", systemImage: "antenna.radiowaves.left.and.right")
            }
            advancedTab.tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .padding()
        .task { await load() }
        // Poll preflight only while reachable so the Advanced tab's Managed tools / Data
        // paths reflect live install state; the bootstrap panel drives the unreachable case.
        .task(id: controller.reachable) {
            guard controller.reachable else { return }
            while !Task.isCancelled {
                await refreshPreflight()
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    // MARK: - Tabs

    private var backendTab: some View {
        Form {
            // On a clean machine the backend isn't up yet — lead with the bootstrap panel
            // (install uv → create venv → install wheel) instead of dead connection fields.
            if !controller.reachable {
                Section("Backend setup") {
                    BootstrapPanel(boot: boot) { Task { await controller.start() } }
                }
            }

            Section("Model backend") {
                Picker("Engine", selection: $modelBackend) {
                    Text("Native MLX (mlx-lm)").tag("mlx")
                    Text("omlx (external server)").tag("omlx")
                }
                .pickerStyle(.radioGroup)
                if let backendKindNote {
                    Text(backendKindNote).font(.caption).foregroundStyle(.secondary)
                }
                if controller.canManageBackend {
                    Button("Save & Restart backend") { Task { await saveBackend() } }
                }
                Text("Native MLX runs in-process (default, no extra install). omlx connects "
                    + "to — or spawns — an external `omlx serve`, so omlx must be installed "
                    + "(e.g. Homebrew). Restart to apply.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            Section("Backend connection") {
                fieldRow("Backend URL", placeholder: "http://127.0.0.1:9981", text: $draft)
                Button("Apply & Reconnect") {
                    var url = draft.trimmingCharacters(in: .whitespaces)
                    // 0.0.0.0 is a *bind* address (Server bind below), not something you
                    // can connect to (it fails with -1022). Be forgiving: rewrite to
                    // localhost so a mistaken paste here doesn't strand the app offline.
                    url = url.replacingOccurrences(of: "://0.0.0.0", with: "://127.0.0.1")
                    draft = url
                    controller.baseURLString = url
                    Task { await controller.refresh() }
                }
                Text("Where the app connects (use 127.0.0.1). To expose the backend on "
                    + "your LAN, set the bind Host to 0.0.0.0 under Server bind below.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            Section("Server bind") {
                HStack(spacing: 10) {
                    Text("Host").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("", text: $backendHost, prompt: Text("127.0.0.1"))
                        .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                    Text("Port").foregroundStyle(.secondary)
                    TextField("", text: $backendPort, prompt: Text("9981"))
                        .textFieldStyle(.roundedBorder).frame(width: 80)
                }
                if let bindNote {
                    Text(bindNote).font(.caption).foregroundStyle(.secondary)
                }
                if controller.canManageBackend {
                    Button("Save & Restart backend") { Task { await saveBind() } }
                }
                Text("Host 0.0.0.0 exposes the backend on your LAN; the app still connects "
                    + "over 127.0.0.1. Changing the port reconnects locally. Restart to apply.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            Section("Backend runtime") {
                // The two rows that answer "where is my server and its environment?" —
                // always accurate, derived from what's actually launched.
                resolvedRow("Backend executable", value: backendExe)
                resolvedRow("Environment (venv)", value: inUseVenv)
                // uv / Python only matter when bootstrapping a managed venv on a clean
                // (thin-app) install — irrelevant to a dev checkout, so keep them folded
                // away rather than cluttering the common case.
                DisclosureGroup("Advanced — packaged-install setup") {
                    pathRow("Managed venv", placeholder: resolvedVenv, text: $venvPath, directories: true)
                    pathRow("uv", placeholder: resolvedUv, text: $uvPath, directories: false)
                    HStack(spacing: 10) {
                        Text("Python version").foregroundStyle(.secondary)
                            .frame(width: labelWidth, alignment: .leading)
                        TextField("", text: $pythonVersion, prompt: Text("3.12"))
                            .textFieldStyle(.roundedBorder).frame(width: 80)
                        Spacer()
                    }
                    Text("These bootstrap the managed venv on a clean (thin-app) install. "
                        + "A dev checkout runs from its own .venv — shown above. Blank = "
                        + "auto-detect.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var agentTab: some View {
        Form {
            Section("Agent") {
                HStack(spacing: 10) {
                    Text("Max output tokens").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("", text: $maxOutputTokens, prompt: Text("4096"))
                        .textFieldStyle(.roundedBorder).frame(width: 100)
                    Button("Save") { Task { await saveMaxOutputTokens() } }
                }
                if let agentNote {
                    Text(agentNote).font(.caption).foregroundStyle(.secondary)
                }
                Text("The ceiling on tokens generated per reply (the model still stops early "
                    + "at its end-of-text). Raising it prevents long answers being cut off. "
                    + "Applies to the next reply — no restart.")
                    .font(.caption2).foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    Text("Max tool steps").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("", text: $maxToolIters, prompt: Text("16"))
                        .textFieldStyle(.roundedBorder).frame(width: 100)
                    Button("Save") { Task { await saveMaxToolIters() } }
                }
                Text("How many tool calls the agent may chain in one turn before stopping. "
                    + "Multi-step tasks (debug → read → fix → test) need more than a few; raise it "
                    + "if complex turns get cut off. Applies to the next turn — no restart.")
                    .font(.caption2).foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    Text("Turn timeout (s)").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("", text: $turnTimeoutS, prompt: Text("off"))
                        .textFieldStyle(.roundedBorder).frame(width: 100)
                    Button("Save") { Task { await saveTurnTimeout() } }
                }
                Text("Wall-clock budget for one turn, in seconds — a runaway turn is stopped "
                    + "between tool steps. Empty or 0 = no limit (default; a legitimately slow "
                    + "large-model turn is never killed). Note: a single in-flight generation isn't "
                    + "interrupted — that's bounded by max output tokens. No restart.")
                    .font(.caption2).foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    Text("Memory ceiling (GB)").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("", text: $memCeilingGb, prompt: Text("off"))
                        .textFieldStyle(.roundedBorder).frame(width: 100)
                    Button("Save") { Task { await saveMemCeiling() } }
                }
                Text("Cap on the memory loaded models may hold. Loading a model that won't fit is "
                    + "refused with a clear error instead of crashing the backend with an "
                    + "out-of-memory. Empty or 0 = no cap (default). Governs chat/vision models; the "
                    + "image/video/audio backends load separately. Enforced on the next load — no "
                    + "restart.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            fusionSection
        }
        .formStyle(.grouped)
        .task { await loadFusion() }
    }

    /// Chat models eligible for a Fusion panel/judge — excludes embedding/image/video kinds and
    /// the virtual "fusion" model itself.
    private var fusionCandidates: [String] {
        controller.models
            .filter { !["embedding", "image", "video"].contains($0.type ?? "") && $0.id != "fusion" }
            .map(\.id)
    }

    private var fusionSection: some View {
        Section("Fusion (panel + judge)") {
            Toggle("Enable Fusion", isOn: $fusionEnabled)
                .onChange(of: fusionEnabled) { _, _ in Task { await saveFusion() } }
            if fusionCandidates.isEmpty {
                Text("No chat models available yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Panel — each model answers, then the judge synthesizes one answer:")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(fusionCandidates, id: \.self) { id in
                    Toggle(isOn: Binding(
                        get: { fusionPanel.contains(id) },
                        set: { on in
                            if on { fusionPanel.insert(id) } else { fusionPanel.remove(id) }
                            Task { await saveFusion() }
                        }
                    )) { Text(id).font(.callout) }
                }
                Picker("Judge", selection: $fusionJudge) {
                    Text("None").tag(String?.none)
                    ForEach(fusionCandidates, id: \.self) { Text($0).tag(Optional($0)) }
                }
                .onChange(of: fusionJudge) { _, _ in Task { await saveFusion() } }
            }
            Text("Then pick the “fusion” model in Chat or Models to run a panel turn. "
                + "Models load one at a time, so it's slower but cross-checked.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func loadFusion() async {
        guard let cfg = try? await controller.client.fusion() else { return }
        fusionEnabled = cfg.enabled
        fusionPanel = Set(cfg.panel)
        fusionJudge = cfg.judge
    }

    private func saveFusion() async {
        try? await controller.client.setFusion(
            enabled: fusionEnabled, panel: Array(fusionPanel), judge: fusionJudge ?? ""
        )
    }

    private var gatewaysTab: some View {
        Form {
            Section("Gateways") {
                HStack {
                    Text("Telegram").font(.subheadline).bold()
                    Spacer()
                    gatewayStatusBadge()
                }
                HStack(spacing: 10) {
                    Text("Bot token").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    SecureField(
                        telegramConfigured ? (telegramTokenMasked ?? "set") : "123456:ABC-DEF…",
                        text: $telegramTokenInput
                    )
                    .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                }
                HStack(spacing: 10) {
                    Text("Allowed user IDs").foregroundStyle(.secondary)
                        .frame(width: labelWidth, alignment: .leading)
                    TextField("comma-separated, e.g. 12345, 67890", text: $telegramAllowlist)
                        .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                }
                HStack {
                    Button("Save & (re)start") { Task { await saveGateways() } }
                    if telegramConfigured {
                        Button("Disable") { Task { await disableTelegram() } }
                    }
                }
                if let gatewaysNote {
                    Text(gatewaysNote).font(.caption).foregroundStyle(.secondary)
                }
                Text("Applies live — the gateway (re)starts on save, no backend restart. The "
                    + "token is stored locally and never shown in full; leave it blank to keep "
                    + "the current one. The allowlist is deny-by-default (empty = nobody).")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private var advancedTab: some View {
        Form {
            managedTools

            Section("Model paths") {
                // Discovery is a filesystem scan, so every change here applies live — no
                // Save/Restart. Editing a path commits on Return or via Choose…
                modelPathRow("Models directory", text: $modelsDir)
                modelPathRow("Download directory", text: $downloadDir)

                // Only models under these dirs are listed. Add more to scan several
                // collections without moving them.
                VStack(alignment: .leading, spacing: 6) {
                    Text("Additional model directories").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    ForEach(Array(extraModelDirs.enumerated()), id: \.offset) { idx, dir in
                        HStack(spacing: 8) {
                            Text(dir).font(.callout.monospaced()).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Button(role: .destructive) {
                                extraModelDirs.remove(at: idx)
                                Task { await applyModelPaths() }
                            } label: { Image(systemName: "minus.circle") }
                                .buttonStyle(.borderless)
                        }
                    }
                    Button("Add directory…") { addExtraModelDir() }
                }

                // Custom binding so the toggle only applies on user action, not when
                // load() seeds the state.
                Toggle("Also list models from the HuggingFace cache", isOn: Binding(
                    get: { hfCache },
                    set: { hfCache = $0; Task { await applyModelPaths() } }
                ))

                if let savedNote {
                    Text(savedNote).font(.caption).foregroundStyle(.secondary)
                }
                Text("Changes apply immediately — only models inside these directories are "
                    + "listed (the HF cache is off unless enabled above). Paths are "
                    + "repointed only; files aren't moved. Config: \(configPath)")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            dataPaths
        }
        .formStyle(.grouped)
    }

    // MARK: - Managed tools + data paths (absorbed from the former Setup tab)

    private func refreshPreflight() async {
        preflight = try? await controller.client.preflight()
    }

    @ViewBuilder
    private var managedTools: some View {
        if controller.reachable, let r = preflight {
            Section {
                ForEach(r.tools) { tool in
                    HStack {
                        Image(systemName: tool.installed ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(tool.installed ? .green : .secondary)
                        VStack(alignment: .leading) {
                            Text(tool.label)
                            Text(tool.version.map { "\(tool.package) · \($0)" } ?? tool.package)
                                .font(.caption).foregroundStyle(.secondary)
                                .help(tool.source ?? "")
                        }
                        Spacer()
                        toolAction(tool, install: r.installs.first { $0.feature == tool.feature })
                    }
                }
            } header: {
                HStack {
                    Text("Managed tools")
                    Spacer()
                    Button("Run setup wizard") { controller.showFirstRun = true }
                        .font(.caption).buttonStyle(.borderless)
                }
            }
        }
    }

    @ViewBuilder
    private func toolAction(_ tool: ToolCheckDTO, install: InstallDTO?) -> some View {
        switch install?.status {
        case "installing":
            ProgressView().controlSize(.small)
        case "error":
            VStack(alignment: .trailing) {
                // Re-running covers both a failed first install and a failed upgrade;
                // `upgrade` only matters once the tool is already present.
                Button("Retry") {
                    Task { try? await controller.client.installTool(feature: tool.feature, upgrade: tool.installed) }
                }
                Text(install?.error ?? "failed").font(.caption2).foregroundStyle(.red).lineLimit(1)
            }
        default:
            if tool.installed {
                if install?.status == "done", controller.canManageBackend {
                    Button("Restart to use") { Task { await controller.restart() } }
                } else if tool.updateAvailable {
                    // Only offer an update when a newer version actually exists (N5), so a
                    // satisfied tool doesn't show a misleading always-on "更新套件".
                    Button("更新套件") {
                        Task { try? await controller.client.installTool(feature: tool.feature, upgrade: true) }
                    }
                } else {
                    Label("Installed", systemImage: "checkmark")
                        .labelStyle(.titleOnly).foregroundStyle(.secondary).font(.caption)
                }
            } else {
                Button("Install") {
                    Task { try? await controller.client.installTool(feature: tool.feature) }
                }
            }
        }
    }

    @ViewBuilder
    private var dataPaths: some View {
        if controller.reachable, let r = preflight {
            Section("Data paths (XDG)") {
                ForEach(r.paths) { p in
                    HStack {
                        Image(systemName: p.exists ? "checkmark.circle.fill" : "circle.dashed")
                            .foregroundStyle(p.exists ? .green : .secondary)
                        Text(p.name).frame(width: 70, alignment: .leading)
                        Text(p.path).font(.caption).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
    }

    // MARK: - Row helpers (single leading label — no duplicated Form labels)

    /// A labelled text field: one leading label, one editable field. `placeholder`
    /// shows the in-use/default value in grey when the field is blank.
    @ViewBuilder
    private func fieldRow(_ label: String, placeholder: String, text: Binding<String>) -> some View {
        HStack(spacing: 10) {
            Text(label).foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
            TextField("", text: text, prompt: Text(placeholder))
                .textFieldStyle(.roundedBorder).autocorrectionDisabled()
        }
    }

    /// A model-path row that commits live: Return or a changed Choose… triggers an
    /// immediate apply (no Save button). Distinct from `pathRow`, which edits
    /// UserDefaults-backed bootstrap paths that shouldn't auto-PUT config.
    @ViewBuilder
    private func modelPathRow(_ label: String, text: Binding<String>) -> some View {
        HStack(spacing: 10) {
            Text(label).foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
            TextField("", text: text, prompt: Text(""))
                .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                .onSubmit { Task { await applyModelPaths() } }
            Button("Choose…") {
                let before = text.wrappedValue
                choose(into: text, directories: true)
                if text.wrappedValue != before { Task { await applyModelPaths() } }
            }
        }
    }

    /// Like `fieldRow` but with a trailing Choose… picker (file or folder).
    @ViewBuilder
    private func pathRow(
        _ label: String, placeholder: String, text: Binding<String>, directories: Bool
    ) -> some View {
        HStack(spacing: 10) {
            Text(label).foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
            TextField("", text: text, prompt: Text(placeholder))
                .textFieldStyle(.roundedBorder).autocorrectionDisabled()
            Button("Choose…") { choose(into: text, directories: directories) }
        }
    }

    /// A read-only, selectable path display for a value the user can't set directly
    /// (it's derived) — e.g. the backend executable resolved from the venv.
    @ViewBuilder
    private func resolvedRow(_ label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(label).foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
            Text(value.isEmpty ? "—" : value)
                .font(.callout.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func choose(into binding: Binding<String>, directories: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = directories
        panel.canChooseFiles = !directories
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = directories
        panel.prompt = "Choose"
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespaces)
        if !current.isEmpty {
            let url = URL(fileURLWithPath: current)
            panel.directoryURL = directories ? url : url.deletingLastPathComponent()
        }
        if panel.runModal() == .OK, let url = panel.url {
            binding.wrappedValue = url.path
        }
    }

    // MARK: - Backend actions

    private func saveBind() async {
        let host = backendHost.trimmingCharacters(in: .whitespaces)
        guard let port = Int(backendPort.trimmingCharacters(in: .whitespaces)),
            (1...65535).contains(port)
        else {
            bindNote = "Port must be a number 1–65535."
            return
        }
        do {
            try await controller.client.putConfig(
                backendHost: host.isEmpty ? nil : host, backendPort: port
            )
            // The app always reaches the backend over localhost; follow the new port.
            controller.baseURLString = "http://127.0.0.1:\(port)"
            draft = controller.baseURLString
            bindNote = "Saved — restarting on \(host.isEmpty ? "127.0.0.1" : host):\(port)…"
            await controller.restart()
            bindNote = controller.reachable
                ? "Backend restarted on port \(port)."
                : "Restarted, but not reachable yet — check the Status tab."
        } catch {
            bindNote = "Save failed: \(error)"
        }
    }

    private func load() async {
        draft = controller.baseURLString
        // Resolve what's actually in use so the runtime section shows real paths.
        backendExe = BackendProcess.defaultCommand()?.first
            ?? "(connect-only — app isn't managing a backend)"
        // The venv in use is the parent of <venv>/bin/assistant-server — derive it so the
        // row always reflects reality (dev .venv or managed venv), not a guessed default.
        let serverSuffix = "/bin/assistant-server"
        inUseVenv = backendExe.hasSuffix(serverSuffix)
            ? String(backendExe.dropLast(serverSuffix.count))
            : "— (no managed backend resolved)"
        resolvedUv = Bootstrap.resolveUv() ?? "not found — install uv or set a path"
        // The managed venv only exists on a clean (thin-app) install. A dev checkout runs
        // from its own .venv (the executable above), so this path is intentionally absent
        // — spell that out so an empty/non-existent path doesn't read as "broken".
        let managedVenv = Bootstrap.managedVenvDir().path
        if FileManager.default.fileExists(atPath: managedVenv) {
            resolvedVenv = managedVenv
        } else if backendExe.contains("/.venv/bin/") {
            resolvedVenv = "not used — running from dev .venv above"
        } else {
            resolvedVenv = "\(managedVenv) — not created (run the Setup wizard)"
        }
        if let cfg = try? await controller.client.getConfig() {
            modelsDir = cfg.modelsDir
            downloadDir = cfg.downloadDir
            extraModelDirs = cfg.extraModelDirs
            hfCache = cfg.hfCache
            backendHost = cfg.backendHost
            backendPort = String(cfg.backendPort)
            modelBackend = cfg.modelBackend
            maxOutputTokens = String(cfg.maxOutputTokens)
            maxToolIters = String(cfg.maxToolIters)
            // nil/0 = off → blank field; whole seconds shown without a trailing ".0".
            if let t = cfg.turnTimeoutS, t > 0 {
                turnTimeoutS = t == t.rounded() ? String(Int(t)) : String(t)
            } else {
                turnTimeoutS = ""
            }
            if let g = cfg.memCeilingGb, g > 0 {
                memCeilingGb = g == g.rounded() ? String(Int(g)) : String(g)
            } else {
                memCeilingGb = ""
            }
            telegramAllowlist = cfg.telegramAllowedUsers.map(String.init).joined(separator: ", ")
            telegramConfigured = cfg.telegramConfigured
            telegramTokenMasked = cfg.telegramTokenMasked
            telegramRunning = cfg.telegramRunning
            telegramError = cfg.telegramError
            configPath = cfg.configPath
        }
    }

    private func saveGateways() async {
        let ids = Self.parseAllowlist(telegramAllowlist)
        let token = telegramTokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            // Blank token field = keep the current token; only send it when the user typed one.
            try await controller.client.putConfig(
                telegramToken: token.isEmpty ? nil : token, telegramAllowedUsers: ids
            )
            telegramTokenInput = ""  // never keep the secret sitting in the field
            await load()  // refresh status (running / error / masked)
            gatewaysNote = telegramRunning
                ? "Telegram gateway running."
                : (telegramError.map { "Not running: \($0)" } ?? "Saved.")
        } catch {
            gatewaysNote = "Save failed: \(error)"
        }
    }

    private func disableTelegram() async {
        do {
            try await controller.client.putConfig(telegramToken: "")  // "" clears the token
            telegramTokenInput = ""
            await load()
            gatewaysNote = "Telegram disabled."
        } catch {
            gatewaysNote = "Disable failed: \(error)"
        }
    }

    static func parseAllowlist(_ s: String) -> [Int] {
        s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    @ViewBuilder private func gatewayStatusBadge() -> some View {
        let (label, color): (String, Color) =
            telegramRunning ? ("running", .green)
            : telegramConfigured ? ("stopped", .red)
            : ("off", .gray)
        Text(label)
            .font(.caption)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.2))
            .clipShape(Capsule())
    }

    private func saveMaxOutputTokens() async {
        guard let n = Int(maxOutputTokens.trimmingCharacters(in: .whitespaces)),
            (64...131072).contains(n)
        else {
            agentNote = "Max output tokens must be a number 64–131072."
            return
        }
        do {
            // Applies live — the next reply uses the new ceiling, no backend restart.
            try await controller.client.putConfig(maxOutputTokens: n)
            agentNote = "Saved — replies may now run up to \(n) tokens."
        } catch {
            agentNote = "Save failed: \(error)"
        }
    }

    private func saveMaxToolIters() async {
        guard let n = Int(maxToolIters.trimmingCharacters(in: .whitespaces)),
            (1...100).contains(n)
        else {
            agentNote = "Max tool steps must be a number 1–100."
            return
        }
        do {
            // Applies live — the next turn's loop reads the new budget, no backend restart.
            try await controller.client.putConfig(maxToolIters: n)
            agentNote = "Saved — the agent may now take up to \(n) tool steps per turn."
        } catch {
            agentNote = "Save failed: \(error)"
        }
    }

    private func saveTurnTimeout() async {
        // Empty field = off (send 0, which the backend stores as "no limit").
        let raw = turnTimeoutS.trimmingCharacters(in: .whitespaces)
        let seconds = raw.isEmpty ? 0 : (Double(raw) ?? -1)
        guard (0...86400).contains(seconds) else {
            agentNote = "Turn timeout must be a number 0–86400 seconds (empty or 0 = off)."
            return
        }
        do {
            // Applies live — the next turn reads the new budget, no backend restart.
            try await controller.client.putConfig(turnTimeoutS: seconds)
            agentNote = seconds == 0
                ? "Saved — turn timeout is off (turns may run unbounded)."
                : "Saved — a turn now aborts after \(Int(seconds))s between tool steps."
        } catch {
            agentNote = "Save failed: \(error)"
        }
    }

    private func saveMemCeiling() async {
        // Empty field = no cap (send 0, which the backend stores as None).
        let raw = memCeilingGb.trimmingCharacters(in: .whitespaces)
        let gb = raw.isEmpty ? 0 : (Double(raw) ?? -1)
        guard (0...4096).contains(gb) else {
            agentNote = "Memory ceiling must be a number 0–4096 GB (empty or 0 = no cap)."
            return
        }
        do {
            // Applies live — the next model load enforces the new ceiling, no backend restart.
            try await controller.client.putConfig(memCeilingGb: gb)
            agentNote = gb == 0
                ? "Saved — memory ceiling is off (no admission cap)."
                : "Saved — models over \(Int(gb))GB are refused instead of risking an out-of-memory."
        } catch {
            agentNote = "Save failed: \(error)"
        }
    }

    private func addExtraModelDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Add"
        if panel.runModal() == .OK, let url = panel.url, !extraModelDirs.contains(url.path) {
            extraModelDirs.append(url.path)
            Task { await applyModelPaths() }
        }
    }

    private func saveBackend() async {
        do {
            try await controller.client.putConfig(modelBackend: modelBackend)
            backendKindNote = "Saved — restarting on the \(modelBackend) backend…"
            await controller.restart()
            backendKindNote = controller.reachable
                ? "Now running the \(modelBackend) backend."
                : "Restarted, but not reachable yet — check the Status tab (is omlx installed?)."
        } catch {
            backendKindNote = "Save failed: \(error)"
        }
    }

    /// Push the current model-path fields to the backend and re-scan immediately. The
    /// backend applies discovery changes to the live service (no restart), so the Models
    /// list reflects them right away.
    private func applyModelPaths() async {
        do {
            try await controller.client.putConfig(
                modelsDir: modelsDir.trimmingCharacters(in: .whitespaces),
                downloadDir: downloadDir.trimmingCharacters(in: .whitespaces),
                extraModelDirs: extraModelDirs,
                hfCache: hfCache
            )
            savedNote = "Applied."
            await controller.refresh()
        } catch {
            savedNote = "Apply failed: \(error)"
        }
    }
}

/// Stand up the managed backend venv on a clean machine. Shown inline in Settings when
/// the backend is unreachable (moved here from the former Setup tab, N6).
private struct BootstrapPanel: View {
    @ObservedObject var boot: BackendBootstrap
    var onReady: () -> Void

    var body: some View {
        GroupBox("Backend setup") {
            VStack(alignment: .leading, spacing: 10) {
                switch boot.phase {
                case .unknown:
                    ProgressView().onAppear { boot.detect() }
                case .ready:
                    Label("Backend ready — starting…", systemImage: "checkmark.seal")
                        .onAppear(perform: onReady)
                case .needsUv:
                    Text("uv is required to set up the backend. Install it (writes to "
                        + "~/.local/bin), or set its path under Backend runtime ▸ Advanced.")
                    HStack {
                        Button("Install uv") { Task { await boot.installUv() } }
                        Button("Re-check") { boot.detect() }
                    }
                case .needsBootstrap:
                    Text("Create the managed Python venv (uv-managed Python) and install "
                        + "the bundled backend.")
                    Button("Set up backend") {
                        Task {
                            await boot.bootstrap()
                            if case .ready = boot.phase { onReady() }
                        }
                    }
                case .working(let step):
                    HStack { ProgressView().controlSize(.small); Text(step) }
                case .failed(let why):
                    Label(why, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                    Button("Retry") { boot.detect() }
                }

                if !boot.log.isEmpty {
                    ScrollView {
                        Text(boot.log)
                            .font(.system(.caption2, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 150)
                    .background(Color.black.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
            .padding(6)
        }
    }
}

import AppKit
import SwiftUI

/// First-run setup, shown when config.toml doesn't exist yet. Two steps:
///  1. Runtime — confirm a backend can launch (dev .venv resolves → ready; otherwise
///     install uv and bootstrap the managed venv) and bring it up.
///  2. Model paths — accept the app defaults or point at an existing model directory.
/// Finishing writes config.toml (via `PUT /config`), which clears the first-run gate.
struct FirstRunWizard: View {
    @EnvironmentObject var controller: BackendController
    @StateObject private var boot = BackendBootstrap()

    @State private var step = 1
    @State private var modelsDir = ""
    @State private var downloadDir = ""
    @State private var note: String?
    @State private var recheckNote: String?
    @State private var busy = false
    // Where the managed backend env gets installed (read by `Bootstrap` during bootstrap).
    @AppStorage("venvPath") private var venvPath: String = ""
    // Optional explicit uv path (read by `Bootstrap.resolveUv`) — for a uv the app can't
    // auto-detect (e.g. an unusual install location).
    @AppStorage("uvPath") private var uvPath: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Welcome to Assistant").font(.title2).bold()
                Text("Step \(step) of 2 · \(step == 1 ? "Backend runtime" : "Model paths")")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Divider()

            if step == 1 { runtimeStep } else { pathsStep }

            if let note {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
            Divider()
            footer
        }
        .padding(20)
        .frame(width: 600, height: 460)
    }

    // MARK: - Step 1: runtime

    @ViewBuilder private var runtimeStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            if controller.reachable {
                Label("Backend is ready.", systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                if let exe = BackendProcess.defaultCommand()?.first {
                    Text("Running from:").font(.caption).foregroundStyle(.secondary)
                    Text(exe).font(.callout.monospaced()).textSelection(.enabled)
                }
                Text("Next: choose where models live.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                bootstrapBody
            }
        }
    }

    /// Show the venv-location chooser while we still need to build the environment.
    private var showVenvChooser: Bool {
        switch boot.phase {
        case .needsUv, .needsBootstrap, .failed: return true
        default: return false
        }
    }

    @ViewBuilder private var bootstrapBody: some View {
        if showVenvChooser {
            VStack(alignment: .leading, spacing: 4) {
                Text("Backend environment (venv) — where the managed Python and server "
                    + "get installed:").font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    TextField("", text: $venvPath, prompt: Text(Bootstrap.managedVenvDir().path))
                        .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                    Button("Choose…") { choose(into: $venvPath) }
                }
            }
        }
        switch boot.phase {
        case .unknown:
            ProgressView().onAppear { boot.detect() }
        case .ready:
            // A launch command resolves (dev/managed venv) — just bring the backend up.
            HStack { ProgressView().controlSize(.small); Text("Starting backend…") }
                .onAppear { Task { await controller.start() } }
        case .needsUv:
            Text("uv is required to set up the backend. If you already have it (Homebrew, "
                + "mise/asdf, cargo…) point to it below and Re-check; otherwise install a "
                + "private copy (kept in the app's own folder, not your ~/.local/bin).")
            HStack(spacing: 8) {
                TextField("", text: $uvPath, prompt: Text("path to uv (optional)"))
                    .textFieldStyle(.roundedBorder).autocorrectionDisabled()
                Button("Choose…") { chooseFile(into: $uvPath) }
            }
            HStack {
                Button("Install uv") { Task { await boot.installUv() } }
                Button("Re-check") { recheck() }
            }
            if let recheckNote {
                Label(recheckNote, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
        case .needsBootstrap:
            Text("Create the managed Python environment and install the backend.")
            Button("Set up backend") {
                Task {
                    await boot.bootstrap()
                    if case .ready = boot.phase { await controller.start() }
                }
            }
        case .working(let stepText):
            HStack { ProgressView().controlSize(.small); Text(stepText) }
        case .failed(let why):
            Label(why, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
            Button("Retry") { recheck() }
        }
        if !boot.log.isEmpty {
            ScrollView {
                Text(boot.log)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: 120)
            .background(Color.black.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    // MARK: - Step 2: model paths

    @ViewBuilder private var pathsStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Where should models live? The defaults work out of the box — change "
                + "them only if you already keep models elsewhere.")
                .font(.callout).foregroundStyle(.secondary)
            pathRow("Models directory", text: $modelsDir)
            pathRow("Download directory", text: $downloadDir)
            Text("Saved to \(Bootstrap.configPath().path)")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .task(id: step) { if step == 2 { await loadDefaults() } }
    }

    private func pathRow(_ label: String, text: Binding<String>) -> some View {
        HStack(spacing: 10) {
            Text(label).foregroundStyle(.secondary).frame(width: 140, alignment: .leading)
            TextField("", text: text).textFieldStyle(.roundedBorder).autocorrectionDisabled()
            Button("Choose…") { choose(into: text) }
        }
    }

    // MARK: - Footer / navigation

    @ViewBuilder private var footer: some View {
        HStack {
            Button("Skip for now") { controller.showFirstRun = false }
                .foregroundStyle(.secondary)
            Spacer()
            if step == 2 {
                Button("Back") { step = 1 }
            }
            if step == 1 {
                Button("Next") { step = 2 }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!controller.reachable)
            } else {
                Button("Finish") { Task { await finish() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy || !controller.reachable)
            }
        }
    }

    // MARK: - Actions

    /// Re-run detection and tell the user the outcome — a silent `detect()` that lands
    /// back on the same phase looks like the button did nothing.
    private func recheck() {
        boot.detect()
        switch boot.phase {
        case .needsUv:
            recheckNote = "Still can't find uv at \(Bootstrap.resolveUv() ?? "any known location"). "
                + "Install it above, or set its path in Settings ▸ Backend runtime, then re-check."
        default:
            recheckNote = nil  // progressed past the blocker; the view now reflects it
        }
    }

    private func loadDefaults() async {
        guard modelsDir.isEmpty || downloadDir.isEmpty else { return }
        if let cfg = try? await controller.client.getConfig() {
            if modelsDir.isEmpty { modelsDir = cfg.modelsDir }
            if downloadDir.isEmpty { downloadDir = cfg.downloadDir }
        }
    }

    private func finish() async {
        busy = true
        defer { busy = false }
        do {
            try await controller.client.putConfig(
                modelsDir: modelsDir.trimmingCharacters(in: .whitespaces),
                downloadDir: downloadDir.trimmingCharacters(in: .whitespaces)
            )
            // Restart so the chosen model dir is actually scanned, then leave the wizard.
            if controller.canManageBackend {
                await controller.restart()
            } else {
                await controller.refresh()
            }
            controller.showFirstRun = false
        } catch {
            note = "Couldn't save: \(error)"
        }
    }

    private func choose(into binding: Binding<String>) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        let current = binding.wrappedValue.trimmingCharacters(in: .whitespaces)
        if !current.isEmpty { panel.directoryURL = URL(fileURLWithPath: current) }
        if panel.runModal() == .OK, let url = panel.url {
            binding.wrappedValue = url.path
        }
    }

    /// Pick an executable (e.g. uv). Shows hidden files so dotfile dirs like
    /// ~/.local/share/mise/shims are reachable.
    private func chooseFile(into binding: Binding<String>) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.showsHiddenFiles = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            binding.wrappedValue = url.path
        }
    }
}

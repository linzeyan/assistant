import CryptoKit
import Foundation

/// Paths + resolution for the thin-app / managed-venv backend. All user-overridable
/// knobs live in UserDefaults so Settings can edit them.
enum Bootstrap {
    static func dataHome() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let xdg = env["XDG_DATA_HOME"], !xdg.isEmpty {
            return URL(fileURLWithPath: xdg)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/share")
    }

    /// Mirrors the backend's XDG config resolution (config.py `_xdg_dir`) so the app can
    /// tell whether first-run init has happened *without* needing the backend running.
    static func configHome() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let xdg = env["XDG_CONFIG_HOME"], !xdg.isEmpty {
            return URL(fileURLWithPath: xdg)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".config")
    }

    static func configPath() -> URL {
        configHome().appendingPathComponent("assistant/config.toml")
    }

    /// First run is defined by the absence of config.toml — the backend creates it lazily
    /// on the first `PUT /config`, so "no file yet" means the user has never been set up.
    static func configExists() -> Bool {
        FileManager.default.fileExists(atPath: configPath().path)
    }

    private static func expanded(_ key: String) -> String? {
        let raw = UserDefaults.standard.string(forKey: key) ?? ""
        return raw.isEmpty ? nil : (raw as NSString).expandingTildeInPath
    }

    /// Where the managed backend venv lives (override key: `venvPath`).
    static func managedVenvDir() -> URL {
        if let override = expanded("venvPath") { return URL(fileURLWithPath: override) }
        return dataHome().appendingPathComponent("assistant/venv")
    }

    /// App-private bin for tools the app installs itself (uv). Kept out of the user's
    /// ~/.local/bin so a managed install never pollutes their PATH.
    static func appPrivateBin() -> URL {
        dataHome().appendingPathComponent("assistant/bin")
    }

    /// Where the backend's logs live — the same `logs/` dir the backend writes its own
    /// rotating `backend.log` into, so a spawned backend's raw stdout/stderr sits beside it.
    static func logsDir() -> URL {
        dataHome().appendingPathComponent("assistant/logs")
    }

    static func managedServer() -> URL {
        managedVenvDir().appendingPathComponent("bin/assistant-server")
    }

    static func pythonVersion() -> String {
        let v = UserDefaults.standard.string(forKey: "pythonVersion") ?? ""
        return v.isEmpty ? "3.12" : v
    }

    private static func bundledBackendDir() -> URL? {
        Bundle.main.resourceURL?.appendingPathComponent("backend")
    }

    static func bundledWheel() -> URL? {
        guard let dir = bundledBackendDir() else { return nil }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        return entries.first { $0.pathExtension == "whl" }
    }

    static func bundledBootstrapScript() -> URL? {
        bundledBackendDir()?.appendingPathComponent("bootstrap.sh")
    }

    /// Where bootstrap.sh records the hash of the wheel it installed.
    static func managedWheelMarker() -> URL {
        managedVenvDir().appendingPathComponent(".assistant-wheel-sha")
    }

    /// SHA-256 (lowercase hex) of the bundled wheel — matches what bootstrap.sh writes
    /// (`shasum -a 256`), so the app can compare what's shipped vs what's installed.
    static func bundledWheelSHA() -> String? {
        guard let wheel = bundledWheel(), let data = try? Data(contentsOf: wheel) else {
            return nil
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// True when a managed venv exists but was installed from a different wheel than the
    /// one now bundled (the .app was updated, or the venv predates hash tracking) — so its
    /// backend code is stale and must be reinstalled before use.
    static func managedVenvNeedsUpdate() -> Bool {
        guard FileManager.default.isExecutableFile(atPath: managedServer().path) else {
            return false  // no managed venv yet → that's first-run bootstrap, not an update
        }
        guard let bundled = bundledWheelSHA() else { return false }  // dev run: no wheel
        let installed = (try? String(contentsOf: managedWheelMarker(), encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return installed != bundled
    }

    /// Locate uv: explicit override, then common install locations, then a login-shell
    /// lookup (a GUI app's PATH is minimal, but `bash -lc` loads the user's profile so
    /// version managers like mise / asdf / Homebrew resolve).
    static func resolveUv() -> String? {
        let fm = FileManager.default
        if let override = expanded("uvPath"), fm.isExecutableFile(atPath: override) {
            return override
        }
        let home = fm.homeDirectoryForCurrentUser.path
        let candidates = [
            appPrivateBin().appendingPathComponent("uv").path,  // the app's own install
            "\(home)/.local/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
            "\(home)/.cargo/bin/uv",
            // Version-manager shims (mise/asdf) — these dispatch correctly even when run
            // from the GUI's minimal environment, so they're safe to hand to bootstrap.
            "\(home)/.local/share/mise/shims/uv",
            "\(home)/.asdf/shims/uv",
        ]
        if let known = candidates.first(where: { fm.isExecutableFile(atPath: $0) }) {
            return known
        }
        if let viaShell = loginShellLookup("uv"), fm.isExecutableFile(atPath: viaShell) {
            return viaShell
        }
        return nil
    }

    /// `command -v <tool>` inside the user's *real* login+interactive shell, so PATH
    /// reflects their profile — including version managers (mise/asdf) that activate in
    /// the interactive rc (~/.zshrc). The previous bash-only lookup missed those.
    private static func loginShellLookup(_ tool: String) -> String? {
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/bash"
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: shell)
        // -i (interactive) sources ~/.zshrc where mise/asdf activate; -l (login) covers
        // profile-based setups. -c runs the command and exits.
        proc.arguments = ["-ilc", "command -v \(tool)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        let path = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (path?.isEmpty == false) ? path : nil
    }
}

/// Drives first-run setup of the managed backend venv: ensure uv (offer to install
/// it with the user's consent), then run the bundled bootstrap script to create the
/// venv and install the backend wheel.
@MainActor
final class BackendBootstrap: ObservableObject {
    enum Phase: Equatable {
        case unknown
        case ready          // a usable backend command already resolves
        case needsUv        // uv missing; offer install or a path override
        case needsBootstrap // uv present, venv not built yet
        case working(String)
        case failed(String)
    }

    @Published var phase: Phase = .unknown
    @Published var log: String = ""

    func detect() {
        if BackendProcess.defaultCommand() != nil {
            phase = .ready
        } else if Bootstrap.resolveUv() == nil {
            phase = .needsUv
        } else {
            phase = .needsBootstrap
        }
    }

    /// Install uv via its official script (only after explicit user consent). Installs
    /// into the app's private bin (UV_INSTALL_DIR) rather than ~/.local/bin, so we never
    /// touch the user's PATH — resolveUv() looks there first.
    func installUv() async {
        log = ""
        phase = .working("Installing uv…")
        let dir = Bootstrap.appPrivateBin().path
        let script = "mkdir -p \"$UV_INSTALL_DIR\" && curl -LsSf https://astral.sh/uv/install.sh | sh"
        let code = await run("/bin/bash", ["-lc", script], env: ["UV_INSTALL_DIR": dir])
        if code != 0 {
            phase = .failed("uv install exited \(code)")
        } else if Bootstrap.resolveUv() == nil {
            phase = .failed("uv installed but not found — set its path in Settings")
        } else {
            phase = .needsBootstrap
        }
    }

    /// Create the venv and install the bundled wheel. On success the managed
    /// `assistant-server` exists and `BackendProcess.defaultCommand()` will find it.
    func bootstrap() async {
        guard let uv = Bootstrap.resolveUv() else { phase = .needsUv; return }
        guard let wheel = Bootstrap.bundledWheel(), let script = Bootstrap.bundledBootstrapScript() else {
            phase = .failed("bundled backend payload missing (build with `make app-package`)")
            return
        }
        log = ""
        phase = .working("Creating venv + installing backend…")
        let code = await runBootstrap(uv: uv, wheel: wheel, script: script)
        phase = code == 0 ? .ready : .failed("bootstrap exited \(code)")
    }

    /// Reinstall the bundled wheel into an existing managed venv (the app-update path).
    /// bootstrap.sh is idempotent — it reuses the venv and force-reinstalls just the
    /// assistant package — so this is a fast refresh. Best effort: returns success.
    func updateManagedVenv() async -> Bool {
        guard let uv = Bootstrap.resolveUv(),
              let wheel = Bootstrap.bundledWheel(),
              let script = Bootstrap.bundledBootstrapScript()
        else { return false }
        log = ""
        phase = .working("Updating backend…")
        let code = await runBootstrap(uv: uv, wheel: wheel, script: script)
        phase = code == 0 ? .ready : .failed("update exited \(code)")
        return code == 0
    }

    private func runBootstrap(uv: String, wheel: URL, script: URL) async -> Int32 {
        await run("/bin/bash", [script.path], env: [
            "UV": uv,
            "VENV_DIR": Bootstrap.managedVenvDir().path,
            "WHEEL": wheel.path,
            "PYTHON_VERSION": Bootstrap.pythonVersion(),
        ])
    }

    /// Run a process, streaming combined stdout+stderr into `log`.
    private func run(_ launch: String, _ args: [String], env: [String: String]) async -> Int32 {
        await withCheckedContinuation { (cont: CheckedContinuation<Int32, Never>) in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: launch)
            proc.arguments = args
            var merged = ProcessInfo.processInfo.environment
            for (k, v) in env { merged[k] = v }
            proc.environment = merged

            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor in self.log += text }
            }
            proc.terminationHandler = { p in
                pipe.fileHandleForReading.readabilityHandler = nil
                cont.resume(returning: p.terminationStatus)
            }
            do {
                try proc.run()
            } catch {
                Task { @MainActor in self.log += "\nfailed to launch: \(error)" }
                cont.resume(returning: -1)
            }
        }
    }
}

import Foundation

/// Connect-or-spawn for the Python backend, mirroring how the backend itself manages
/// its model server. By default the app connects to a running `assistant-server`; if a
/// launch command is resolved, it spawns one as a managed child and terminates it on
/// stop. Best-effort: a spawn failure never blocks the UI.
final class BackendProcess {
    private var process: Process?
    let command: [String]?

    init(command: [String]?) {
        self.command = command
    }

    /// Resolve how to launch the backend, or nil to stay connect-only.
    ///
    /// Order: an explicit `ASSISTANT_SERVER` override, then a backend bundled inside
    /// the .app (future venvstacks packaging), then the project's dev venv found by
    /// walking up from the executable. Returning nil is fine — the app then just
    /// connects to whatever is already serving.
    static func defaultCommand() -> [String]? {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        if let override = env["ASSISTANT_SERVER"], !override.isEmpty {
            return [override]
        }
        // A packaged .app is a release artifact: it must be self-contained and must NOT
        // reach into a dev checkout's .venv — even when it happens to sit inside the repo
        // (e.g. dist/Assistant.app). Walking up to find a project venv is purely a
        // `swift run` convenience. ASSISTANT_FORCE_MANAGED forces the same self-contained
        // behaviour in a dev build for testing the real first-run bootstrap.
        let isPackagedApp = Bundle.main.bundleURL.pathExtension == "app"
        let forceManaged = isPackagedApp || !(env["ASSISTANT_FORCE_MANAGED"] ?? "").isEmpty
        if !forceManaged {
            var dir = Bundle.main.bundleURL
            for _ in 0..<8 {
                let candidate = dir.appendingPathComponent(".venv/bin/assistant-server")
                if fm.isExecutableFile(atPath: candidate.path) { return [candidate.path] }
                dir.deleteLastPathComponent()
            }
        }
        // The bootstrapped managed venv (the thin-app target on a clean machine).
        let managed = Bootstrap.managedServer().path
        if fm.isExecutableFile(atPath: managed) { return [managed] }
        return nil
    }

    /// Whether the resolved launch command is the bootstrapped managed venv (not a dev
    /// .venv or an explicit override) — i.e. the install an app update needs to refresh.
    static func usesManagedVenv() -> Bool {
        defaultCommand()?.first == Bootstrap.managedServer().path
    }

    /// Best-effort: terminate whatever is listening on `port`. Used after a backend
    /// update to evict a still-running stale server so the fresh code can be spawned
    /// (connect-or-spawn would otherwise reuse the old process).
    static func killOnPort(_ port: Int) {
        let lsof = Process()
        lsof.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        lsof.arguments = ["-ti", "tcp:\(port)"]
        let pipe = Pipe()
        lsof.standardOutput = pipe
        lsof.standardError = FileHandle.nullDevice
        do { try lsof.run() } catch { return }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        lsof.waitUntilExit()
        let pids = String(data: data, encoding: .utf8)?
            .split(whereSeparator: \.isNewline)
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? []
        for pid in pids { kill(pid_t(pid), SIGTERM) }
    }

    func spawn() {
        guard process == nil, let command, let exe = command.first else { return }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: exe)
        proc.arguments = Array(command.dropFirst())
        // A GUI-spawned backend inherits no usable console, so its stdout/stderr — uvicorn's
        // lifecycle lines and any traceback raised *before* Python logging is configured —
        // would be lost, defeating diagnosis. Tee them to a file beside the backend's own
        // rotating log. Truncated per spawn so it holds just the current run (the backend's
        // backend.log keeps the rotated history); best-effort — nil just inherits the app's.
        if let handle = Self.spawnOutHandle() {
            proc.standardOutput = handle
            proc.standardError = handle
        }
        do {
            try proc.run()
            process = proc
        } catch {
            NSLog("assistant backend spawn failed: \(error)")
        }
    }

    /// A fresh, append-positioned handle to `logs/backend.out.log` for this spawn.
    private static func spawnOutHandle() -> FileHandle? {
        let fm = FileManager.default
        let dir = Bootstrap.logsDir()
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("backend.out.log")
        guard fm.createFile(atPath: url.path, contents: nil) else { return nil }  // truncate
        return try? FileHandle(forWritingTo: url)
    }

    var isRunning: Bool { process?.isRunning ?? false }

    /// Graceful stop: SIGTERM first, then SIGKILL if the backend hasn't exited within `grace`.
    /// uvicorn handles SIGTERM promptly (usually well under 0.5s), so the bounded wait is short
    /// in practice; the escalation guarantees we never leave a wedged backend holding the port
    /// (which would make connect-or-spawn reuse a half-dead process).
    func stop(grace: Double = 2) {
        guard let proc = process else { return }
        process = nil
        guard proc.isRunning else { return }
        proc.terminate()  // SIGTERM
        let deadline = Date().addingTimeInterval(grace)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if proc.isRunning {
            kill(proc.processIdentifier, SIGKILL)  // escalate — it ignored SIGTERM
        }
    }
}

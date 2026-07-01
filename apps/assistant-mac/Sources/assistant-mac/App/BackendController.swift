import AppKit
import Foundation
import SwiftUI

/// The single source of UI truth: backend connection, model list, and selection.
/// `@MainActor` so all `@Published` mutations happen on the main thread.
@MainActor
final class BackendController: ObservableObject {
    // Persisted so a user-changed host/port survives relaunch — otherwise the app would
    // reconnect to the default port next launch while the backend binds the configured one.
    // 0.0.0.0 is a bind address, never a connect target, so sanitise it to localhost on
    // read as well (a value persisted before that guard would otherwise strand us offline).
    @Published var baseURLString: String =
        (UserDefaults.standard.string(forKey: "baseURLString") ?? "http://127.0.0.1:9981")
        .replacingOccurrences(of: "://0.0.0.0", with: "://127.0.0.1")
    {
        didSet { UserDefaults.standard.set(baseURLString, forKey: "baseURLString") }
    }
    @Published var status: StatusDTO?
    @Published var models: [ModelDTO] = []
    /// HTTP backend responds (we can talk to the server at all). Distinct from
    /// `modelBackendReady` — a backend with no model engine (e.g. mlx-lm not installed
    /// yet) is still *reachable*; conflating the two made a healthy server read as offline.
    @Published var reachable: Bool = false
    /// The model engine is available and can list/run models (`/models` reachable).
    @Published var modelBackendReady: Bool = false
    /// True while a start/restart attempt is polling for the backend. Lets the UI show
    /// "starting…" instead of the alarming "not reachable" banner during normal boot.
    @Published var starting: Bool = false
    /// True while reinstalling the managed venv after an app update (slow-ish), so the UI
    /// can say "Updating backend…" rather than a bare "Starting…".
    @Published var updatingBackend: Bool = false
    /// The model the Chat tab is *currently* using — transient, per session. Changing it
    /// from the Chat picker must NOT touch `defaultModel`.
    @Published var selectedModel: String?
    /// The user's persisted default model (set explicitly via the Models tab's "Default").
    /// nil until they choose one, in which case selection falls back to first/loaded.
    @Published var defaultModel: String? =
        UserDefaults.standard.string(forKey: "defaultModel")
    {
        didSet { UserDefaults.standard.set(defaultModel, forKey: "defaultModel") }
    }
    @Published var lastError: String?
    /// Which model a load/unload is in flight for (drives a row spinner), and the last
    /// model-action failure — kept separate from `lastError` so a refresh doesn't wipe it.
    @Published var busyModel: String?
    @Published var modelActionError: String?
    /// Drives the first-run setup wizard. Shown when the user hasn't been initialised
    /// (no config.toml) OR when no backend can even be launched yet (no dev/managed venv
    /// resolves) — the latter is the clean-install case that must bootstrap rather than
    /// sit silently offline. The Setup tab can also re-open it on demand.
    @Published var showFirstRun: Bool =
        !Bootstrap.configExists() || BackendProcess.defaultCommand() == nil

    /// Managed backend: spawned when a launch command resolves (managed/dev venv),
    /// otherwise connect-only. Re-resolved on each start so a just-bootstrapped venv
    /// is picked up without relaunching the app.
    private var backend = BackendProcess(command: BackendProcess.defaultCommand())
    private var terminationObserver: NSObjectProtocol?
    private let bootstrap = BackendBootstrap()
    /// Set around deliberate shutdowns (app quit, user restart) so the health monitor doesn't
    /// fight them by auto-restarting a backend we intentionally stopped.
    private var expectingExit = false
    /// Long-lived supervisor: probes the managed backend and auto-restarts it on a crash.
    private var monitorTask: Task<Void, Never>?

    /// Port the app connects to — followed when reinstalling so we evict the right server.
    private var currentPort: Int { URL(string: baseURLString)?.port ?? 9981 }

    var client: AssistantClient {
        let url = URL(string: baseURLString) ?? URL(string: "http://127.0.0.1:9981")!
        return AssistantClient(baseURL: url)
    }

    func start() async {
        starting = true
        defer { starting = false }
        expectingExit = false  // a fresh start cancels any prior deliberate-stop intent
        // Pick up a managed venv that may have appeared since launch (bootstrap).
        if backend.command == nil, let cmd = BackendProcess.defaultCommand() {
            backend = BackendProcess(command: cmd)
        }
        // App update: the managed venv was installed from an older bundled wheel (or
        // predates hash tracking) → reinstall, then evict any stale running server so the
        // fresh code is spawned rather than reused. Without this, `make app-package` +
        // relaunch keeps serving the old backend.
        if BackendProcess.usesManagedVenv(), Bootstrap.managedVenvNeedsUpdate() {
            updatingBackend = true
            let updated = await bootstrap.updateManagedVenv()
            updatingBackend = false
            if updated {
                backend.stop()
                BackendProcess.killOnPort(currentPort)
                for _ in 0..<20 {  // wait for the old process to release the port
                    if await !probe() { break }
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
            }
        }
        // Connect-or-spawn: only launch a child if nothing is already serving, so a
        // separately-started `make run` backend is reused instead of fighting for the port.
        if await !probe() {
            backend.spawn()
            installTerminationGuard()
        }
        // The child binds the port in ~0.25s now (fast startup), so poll tightly at first to
        // render as soon as it's up, then back off for the long tail (cold venv / slow machine).
        // probe() is a /status GET that fails instantly (connection refused) while it's still
        // booting, so tight polling costs nothing.
        for i in 0..<52 {  // 15×0.1s + 37×0.5s ≈ 20s budget
            if await probe() {
                await refresh()
                startHealthMonitor()
                return
            }
            try? await Task.sleep(nanoseconds: i < 15 ? 100_000_000 : 500_000_000)
        }
        await refresh()  // final attempt surfaces lastError if still down
        startHealthMonitor()
    }

    func stop() {
        expectingExit = true  // deliberate — the monitor must not auto-restart it
        monitorTask?.cancel()
        monitorTask = nil
        backend.stop()
    }

    /// User-initiated backend shutdown from the menu bar — distinct from app Quit: stop the
    /// managed child and suppress the health monitor's auto-restart, but leave the app running.
    /// Frees model RAM and avoids a redundant in-process engine when the user is serving models
    /// elsewhere (e.g. an external `omlx serve`). Reversible via `start()` ("Start Backend").
    func stopBackend() {
        stop()  // SIGTERM the child + cancel auto-restart (sets expectingExit)
        reachable = false  // reflect the deliberate stop immediately, before the next probe
        modelBackendReady = false
    }

    /// Re-evaluate the first-run gate when the app regains focus. macOS keeps the process
    /// alive after the window closes, so the launch-time gate never re-fires — without
    /// this, deleting config.toml and reopening the app wouldn't re-trigger setup. Only
    /// ever opens the wizard (never force-closes it), so an in-progress run isn't disturbed.
    func recheckFirstRun() {
        if !Bootstrap.configExists() || !canManageBackend && !reachable {
            showFirstRun = true
        }
    }

    /// Whether the app can start/stop the backend itself (a launch command resolved).
    var canManageBackend: Bool { backend.command != nil }

    /// Restart the managed backend so a config change / freshly installed tool takes
    /// effect. No-op for an externally-run backend (nothing to restart).
    func restart() async {
        guard canManageBackend else { return }
        starting = true
        defer { starting = false }
        // A deliberate restart wants the backend running afterwards, so clear any stale
        // stop-intent; the `starting` flag keeps the health monitor from racing this.
        expectingExit = false
        backend.stop()
        // Wait for the old process to release the port before respawning. Without this,
        // the new process can't bind and we silently reconnect to the dying instance —
        // which still holds the *old* config, so a path change looked like it needed a
        // full app relaunch to take effect.
        for _ in 0..<20 {  // ~10s
            if await !probe() { break }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        backend.spawn()
        for _ in 0..<40 {
            if await probe() {
                await refresh()
                return
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        await refresh()
    }

    /// Lightweight reachability check used by the boot poll.
    private func probe() async -> Bool {
        (try? await client.status()) != nil
    }

    /// Terminate the spawned backend when the app quits (avoid orphaned children).
    private func installTerminationGuard() {
        guard terminationObserver == nil else { return }
        terminationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.expectingExit = true  // quitting is expected — don't auto-restart
                self?.monitorTask?.cancel()
                self?.backend.stop()
            }
        }
    }

    /// Supervise the managed backend: probe periodically and auto-restart it on a crash, with
    /// exponential backoff so a backend that keeps dying doesn't spin. No-op for an
    /// externally-run backend (nothing we can relaunch). Started once, after the first boot.
    private func startHealthMonitor() {
        guard monitorTask == nil, canManageBackend else { return }
        monitorTask = Task { [weak self] in
            var backoff: UInt64 = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)  // probe cadence
                guard let self, !Task.isCancelled else { return }
                if await self.probe() {
                    backoff = 0  // healthy — reset the backoff
                    continue
                }
                // Unreachable: only act if we own the process and aren't deliberately stopping
                // it (a UI restart already holds `starting`, app quit holds `expectingExit`).
                guard self.canManageBackend, !self.expectingExit, !self.starting else { continue }
                await self.restart()
                if await self.probe() {
                    backoff = 0
                } else {
                    backoff = Self.nextBackoffNanos(backoff)
                    try? await Task.sleep(nanoseconds: backoff)  // back off before retrying
                }
            }
        }
    }

    /// Backoff policy: 1s → 2s → 4s … capped at 30s. Pure (nonisolated), so the schedule is
    /// unit-testable without the main actor.
    nonisolated static func nextBackoffNanos(_ current: UInt64) -> UInt64 {
        let next = current == 0 ? 1_000_000_000 : current * 2
        return min(next, 30_000_000_000)
    }

    func refresh() async {
        do {
            status = try await client.status()
            let result = try await client.models()
            models = result.models
            reachable = true  // the calls succeeded — the HTTP backend is up
            modelBackendReady = result.reachable
            // The default lives on the backend now (shared with Telegram); adopt it so the GUI
            // reflects it. try? keeps an older backend without the route from breaking refresh.
            if let backendDefault = (try? await client.defaultModel())?.model, !backendDefault.isEmpty {
                defaultModel = backendDefault
            }
            if selectedModel == nil || !models.contains(where: { $0.id == selectedModel }) {
                // Prefer the user's persisted default (if still present), then a loaded
                // model, then the first listed.
                let validDefault = defaultModel.flatMap { d in
                    models.first(where: { $0.id == d })?.id
                }
                selectedModel = validDefault
                    ?? models.first(where: { $0.loaded })?.id
                    ?? models.first?.id
            }
            lastError = nil
        } catch is CancellationError {
            // A cancelled request (e.g. the user pressed Stop, which tears down the chat
            // task) is NOT evidence the backend is down — leave reachable as-is so the
            // offline banner doesn't flash on Stop and then stick.
        } catch let urlError as URLError where urlError.code == .cancelled {
            // URLSession reports a cancelled task as -999; same reasoning as above.
        } catch {
            lastError = String(describing: error)
            reachable = false
            modelBackendReady = false
        }
    }

    /// Make `id` the default chat model: update the local UI immediately, switch the current
    /// session to it (existing behavior), and persist to the backend so the Telegram gateway
    /// shares the same default. The local-first order keeps the button feeling instant.
    func setDefaultModel(_ id: String) async {
        defaultModel = id
        selectedModel = id
        do {
            try await client.setDefaultModel(id)
        } catch {
            modelActionError = "Set default failed — \(error)"
        }
    }

    func load(_ id: String) async {
        busyModel = id
        modelActionError = nil
        do {
            try await client.loadModel(id)
        } catch {
            modelActionError = "Load failed — \(error)"
        }
        await refresh()
        busyModel = nil
    }

    func unload(_ id: String) async {
        busyModel = id
        modelActionError = nil
        do {
            try await client.unloadModel(id)
        } catch {
            modelActionError = "Unload failed — \(error)"
        }
        await refresh()
        busyModel = nil
    }

    func delete(_ id: String) async {
        busyModel = id
        modelActionError = nil
        do {
            try await client.deleteModel(id)
            if defaultModel == id { defaultModel = nil }  // don't point default at a gone model
        } catch {
            modelActionError = "Delete failed — \(error)"
        }
        await refresh()
        busyModel = nil
    }
}

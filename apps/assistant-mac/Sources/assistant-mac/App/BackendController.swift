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

    /// Port the app connects to — followed when reinstalling so we evict the right server.
    private var currentPort: Int { URL(string: baseURLString)?.port ?? 9981 }

    var client: AssistantClient {
        let url = URL(string: baseURLString) ?? URL(string: "http://127.0.0.1:9981")!
        return AssistantClient(baseURL: url)
    }

    func start() async {
        starting = true
        defer { starting = false }
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
        // The child takes a moment to bind the port; poll instead of failing on the race.
        for _ in 0..<40 {  // ~20s
            if await probe() {
                await refresh()
                return
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        await refresh()  // final attempt surfaces lastError if still down
    }

    func stop() {
        backend.stop()
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
            MainActor.assumeIsolated { self?.backend.stop() }
        }
    }

    func refresh() async {
        do {
            status = try await client.status()
            let result = try await client.models()
            models = result.models
            reachable = true  // the calls succeeded — the HTTP backend is up
            modelBackendReady = result.reachable
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
        } catch {
            lastError = String(describing: error)
            reachable = false
            modelBackendReady = false
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

import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    // Status first: it's the at-a-glance health view (backend reachable? which model
    // backend? how many models?) — the right landing tab while the platform is young.
    case status = "Status"
    case chat = "Chat"
    case models = "Models"
    case downloads = "Downloads"
    case skills = "Skills"
    case memory = "Memory"
    case settings = "Settings"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .models: return "cpu"
        case .downloads: return "arrow.down.circle"
        case .skills: return "wand.and.stars"
        case .memory: return "tray.full"
        case .status: return "waveform.path.ecg"
        case .settings: return "gearshape"
        }
    }
}

struct RootView: View {
    @EnvironmentObject var controller: BackendController
    @State private var selection: AppSection? = .status

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon).tag(section)
            }
            .navigationSplitViewColumnWidth(180)
        } detail: {
            VStack(spacing: 0) {
                if controller.starting && !controller.reachable {
                    startingBar
                } else if !controller.reachable {
                    offlineBanner
                }
                detail
            }
        }
        // First run (no config.toml yet): gate the app behind the setup wizard.
        .sheet(isPresented: $controller.showFirstRun) {
            FirstRunWizard()
        }
    }

    @ViewBuilder private var detail: some View {
        switch selection ?? .status {
        case .chat: ChatScreen()
        case .models: ModelsScreen()
        case .downloads: DownloadsScreen()
        case .skills: SkillsScreen()
        case .memory: MemoryScreen()
        case .status: StatusScreen()
        case .settings: SettingsScreen()
        }
    }

    /// Shown during the normal boot/restart poll so launch doesn't flash the alarming
    /// "not reachable" banner before the backend has had a chance to come up.
    private var startingBar: some View {
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Text(controller.updatingBackend ? "Updating backend…" : "Starting backend…")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(10)
        .background(Color.secondary.opacity(0.10))
    }

    /// Shown on every screen while the backend is unreachable — the single most
    /// confusing failure mode, so make it actionable rather than silent.
    private var offlineBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "bolt.slash.fill").foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Backend not reachable").bold()
                Text(controller.canManageBackend
                    ? "Retry starts \(controller.baseURLString) automatically."
                    : "Start it with `make run`, or set the URL in Settings.")
                    .font(.caption).foregroundStyle(.secondary)
                if let err = controller.lastError {
                    Text(err).font(.caption2).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer()
            Button("Retry") { Task { await controller.start() } }
                .keyboardShortcut("r", modifiers: .command)
        }
        .padding(10)
        .background(Color.orange.opacity(0.12))
    }
}

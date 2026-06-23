import SwiftUI

struct StatusScreen: View {
    @EnvironmentObject var controller: BackendController
    @State private var preflight: PreflightDTO?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Status").font(.title2).bold()

            GroupBox("Backend") {
                VStack(alignment: .leading, spacing: 6) {
                    row("URL", controller.baseURLString)
                    row("Reachable", controller.reachable ? "yes" : "no")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let backend = controller.status?.omlx {
                GroupBox(modelBackendTitle) {
                    VStack(alignment: .leading, spacing: 6) {
                        row("State", backend.state)
                        row("Reachable", backend.reachable ? "yes" : "no")
                        row("Detail", backend.detail)
                        if let url = backend.baseURL { row("URL", url) }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            if let preflight {
                GroupBox("Runtime") {
                    VStack(alignment: .leading, spacing: 6) {
                        row("Python", preflight.python)
                        row("venv", preflight.venv)
                        row("Config", preflight.configExists
                            ? preflight.configPath
                            : "\(preflight.configPath) (not created yet)")
                        row("Models", "\(preflight.models.count) discovered")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            if let error = controller.lastError {
                Text(error).font(.caption).foregroundStyle(.red)
            }

            Spacer()
            HStack(spacing: 10) {
                Button("Refresh") { Task { await controller.refresh() } }
                if controller.canManageBackend {
                    // Reloads the backend process — needed to pick up new tools / code
                    // after an update (the server imports everything once at startup).
                    Button("Restart backend") { Task { await controller.restart() } }
                        .disabled(controller.starting)
                }
                if controller.starting {
                    ProgressView().controlSize(.small)
                    Text("restarting…").font(.caption).foregroundStyle(.secondary)
                }
            }
            if controller.canManageBackend {
                Text("Restart to load newly added tools or backend changes.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding()
        // Runtime facts (Python/venv/config/models) come from /preflight, the same source
        // the Setup screen uses; poll while reachable so Status is a complete health view.
        .task(id: controller.reachable) {
            guard controller.reachable else { preflight = nil; return }
            while !Task.isCancelled {
                preflight = try? await controller.client.preflight()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    /// Title reflects the configured model backend instead of always saying "omlx" —
    /// the native default is in-process MLX, not the external omlx server.
    private var modelBackendTitle: String {
        switch controller.status?.modelBackend {
        case "mlx": return "Model backend · native MLX"
        case "omlx": return "Model backend · omlx"
        case let other?: return "Model backend · \(other)"
        default: return "Model backend"
        }
    }

    private func row(_ key: String, _ value: String) -> some View {
        HStack {
            Text(key).foregroundStyle(.secondary)
            Spacer()
            Text(value).textSelection(.enabled)
        }
    }
}

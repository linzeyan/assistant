import AppKit
import SwiftUI

struct MenuBarContent: View {
    @EnvironmentObject var controller: BackendController

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                controller.reachable ? "Backend: online" : "Backend: offline",
                systemImage: controller.reachable ? "circle.fill" : "circle"
            )
            .foregroundStyle(controller.reachable ? .green : .secondary)

            if let model = controller.selectedModel {
                Text("Model: \(model)").font(.caption).foregroundStyle(.secondary)
            }

            Divider()

            Button("Refresh") { Task { await controller.refresh() } }
            // Stop/Start the backend without quitting the app, so its model RAM can be freed
            // (e.g. when serving inference from an external omlx). Only when the app owns the
            // process — for a connect-only/external backend there's nothing for us to manage.
            if controller.canManageBackend {
                if controller.reachable {
                    Button("Stop Backend") { controller.stopBackend() }
                } else {
                    Button("Start Backend") { Task { await controller.start() } }
                }
            }
            Button("Quit") {
                controller.stop()
                NSApplication.shared.terminate(nil)
            }
        }
        .padding(12)
        .frame(width: 220)
    }
}

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
            Button("Quit") {
                controller.stop()
                NSApplication.shared.terminate(nil)
            }
        }
        .padding(12)
        .frame(width: 220)
    }
}

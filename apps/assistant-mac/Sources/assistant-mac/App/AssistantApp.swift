import SwiftUI

@main
struct AssistantApp: App {
    @StateObject private var controller = BackendController()
    @StateObject private var chat = ChatModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup("Assistant") {
            RootView()
                .environmentObject(controller)
                .environmentObject(chat)
                .frame(minWidth: 860, minHeight: 580)
                .task { await controller.start() }
        }
        .windowResizability(.contentSize)
        // Re-check setup when the app regains focus so deleting config.toml and
        // reopening re-triggers the wizard (the process survives window close).
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { controller.recheckFirstRun() }
        }

        MenuBarExtra("Assistant", systemImage: "brain.head.profile") {
            MenuBarContent()
                .environmentObject(controller)
        }
        .menuBarExtraStyle(.window)
    }
}

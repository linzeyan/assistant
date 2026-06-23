import SwiftUI

struct DownloadsScreen: View {
    @EnvironmentObject var controller: BackendController
    @State private var repoId = ""
    @State private var downloads: [DownloadDTO] = []
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Downloads").font(.title2).bold()
                Spacer()
                Button { Task { await refresh() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }
            .padding()

            HStack {
                TextField(
                    "HuggingFace repo id (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)",
                    text: $repoId
                )
                .textFieldStyle(.roundedBorder)
                .onSubmit(start)
                Button("Download", action: start)
                    .disabled(repoId.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(.horizontal)

            if let error {
                Text(error).foregroundStyle(.red).font(.caption).padding(.horizontal)
            }

            if downloads.isEmpty {
                ContentUnavailableView(
                    "No downloads",
                    systemImage: "arrow.down.circle",
                    description: Text("Enter a repo id above to fetch a model into the local cache.")
                )
            } else {
                List(downloads) { item in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(item.repoId)
                            if let detail = item.error {
                                Text(detail).font(.caption).foregroundStyle(.red)
                            }
                        }
                        Spacer()
                        statusBadge(item.status)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        // Poll while the screen is visible so in-progress downloads update live.
        .task {
            while !Task.isCancelled {
                await refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func start() {
        let rid = repoId.trimmingCharacters(in: .whitespaces)
        guard !rid.isEmpty else { return }
        Task {
            do {
                try await controller.client.startDownload(repoId: rid)
                repoId = ""
                await refresh()
            } catch {
                self.error = String(describing: error)
            }
        }
    }

    private func refresh() async {
        do {
            downloads = try await controller.client.downloads().downloads
            error = nil
        } catch {
            self.error = String(describing: error)
        }
    }

    @ViewBuilder private func statusBadge(_ status: String) -> some View {
        Text(status)
            .font(.caption)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color(for: status).opacity(0.2))
            .clipShape(Capsule())
    }

    private func color(for status: String) -> Color {
        switch status {
        case "done": return .green
        case "error": return .red
        default: return .orange
        }
    }
}

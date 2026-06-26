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
                    .disabled(repoId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
                // Fill the area so the placeholder centres, matching the Memory screen
                // (without this it's pinned to the top by the outer .top alignment).
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(downloads) { item in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(item.repoId).lineLimit(1)
                            Spacer()
                            statusBadge(item.status)
                            if item.isActive {
                                Button("Cancel") { cancel(item.repoId) }
                                    .buttonStyle(.borderless).font(.caption)
                            } else {
                                if item.isResumable {
                                    Button("Retry") { retry(item.repoId) }
                                        .buttonStyle(.borderless).font(.caption)
                                }
                                // A finished/cancelled/failed entry can be cleared from the list.
                                Button { remove(item.repoId) } label: {
                                    Image(systemName: "xmark.circle")
                                }
                                .buttonStyle(.borderless).foregroundStyle(.secondary)
                                .help("Remove from list")
                            }
                        }
                        if item.isActive {
                            ProgressView(value: item.fraction)  // nil -> indeterminate
                            progressLine(item)
                        }
                        if let detail = item.error {
                            Text(detail).font(.caption).foregroundStyle(.red).lineLimit(2)
                        }
                    }
                    .padding(.vertical, 2)
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
        let rid = repoId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rid.isEmpty else { return }
        // A pasted multi-line or doubled value (e.g. "org/name\norg/name") would otherwise be
        // sent verbatim and rejected by the hub with a confusing error — reject it clearly.
        guard !rid.contains(where: \.isWhitespace) else {
            error = "Repo id must be a single 'namespace/name' with no spaces or line breaks."
            return
        }
        Task {
            do {
                try await controller.client.startDownload(repoId: rid)
                repoId = ""
                error = nil
                await refresh()
            } catch {
                self.error = String(describing: error)
            }
        }
    }

    private func cancel(_ repoId: String) {
        Task {
            do {
                try await controller.client.cancelDownload(repoId: repoId)
                await refresh()
            } catch {
                self.error = String(describing: error)
            }
        }
    }

    private func retry(_ repoId: String) {
        Task {
            do {
                try await controller.client.retryDownload(repoId: repoId)
                await refresh()
            } catch {
                self.error = String(describing: error)
            }
        }
    }

    private func remove(_ repoId: String) {
        Task {
            do {
                try await controller.client.removeDownload(repoId: repoId)
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

    private func progressLine(_ item: DownloadDTO) -> some View {
        Text(Self.progressText(item))
            .font(.caption2).foregroundStyle(.secondary)
    }

    /// "123 MB / 456 MB · 27% · ETA 2m 30s" — omitting parts that aren't known yet.
    static func progressText(_ item: DownloadDTO) -> String {
        var parts: [String] = []
        if item.totalBytes > 0 {
            parts.append("\(bytes(item.downloadedBytes)) / \(bytes(item.totalBytes))")
            if let f = item.fraction { parts.append("\(Int(f * 100))%") }
        } else if item.downloadedBytes > 0 {
            parts.append(bytes(item.downloadedBytes))
        }
        if let secs = item.etaSeconds { parts.append("ETA \(eta(secs))") }
        return parts.joined(separator: " · ")
    }

    static func bytes(_ n: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(n), countStyle: .file)
    }

    static func eta(_ seconds: Int) -> String {
        if seconds >= 3600 { return "\(seconds / 3600)h \((seconds % 3600) / 60)m" }
        if seconds >= 60 { return "\(seconds / 60)m \(seconds % 60)s" }
        return "\(seconds)s"
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
        case "cancelled": return .gray
        default: return .orange
        }
    }
}

import AppKit
import SwiftUI

struct ModelsScreen: View {
    @EnvironmentObject var controller: BackendController
    @State private var pendingDelete: ModelDTO?
    // The id whose copy icon currently shows the "copied" checkmark; cleared after a
    // short delay so a click registers as visible feedback (the copy is otherwise silent).
    @State private var copiedID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Models").font(.title2).bold()
                Spacer()
                Button { Task { await controller.refresh() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }
            .padding()

            if let err = controller.modelActionError {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
                    Text(err).font(.callout)
                    Spacer()
                    Button("Dismiss") { controller.modelActionError = nil }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                }
                .padding(10)
                .background(Color.yellow.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal).padding(.bottom, 8)
            }

            if !controller.reachable {
                ContentUnavailableView(
                    "Backend not reachable",
                    systemImage: "bolt.slash",
                    description: Text(controller.status?.omlx.detail ?? "Start the backend (make run).")
                )
            } else if controller.models.isEmpty {
                ContentUnavailableView(
                    "No models",
                    systemImage: "tray",
                    description: Text("Fetch one from the Downloads tab, or drop weights in the models dir.")
                )
            } else {
                List(controller.models) { model in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(model.id)
                                Button {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(model.id, forType: .string)
                                    copiedID = model.id
                                    Task {
                                        try? await Task.sleep(nanoseconds: 1_200_000_000)
                                        if copiedID == model.id { copiedID = nil }
                                    }
                                } label: {
                                    Image(systemName: copiedID == model.id
                                        ? "checkmark.circle.fill" : "doc.on.doc")
                                }
                                    .buttonStyle(.borderless)
                                    .foregroundStyle(copiedID == model.id ? Color.green : Color.secondary)
                                    .help(copiedID == model.id ? "Copied!" : "Copy model name")
                            }
                            // kind + on-disk size (the actionable facts). Provenance is
                            // shown only for cached models, where it explains the origin.
                            Text(subtitle(model)).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if model.id == controller.defaultModel {
                            Label("default", systemImage: "star.fill")
                                .labelStyle(.titleAndIcon)
                                .font(.caption2).foregroundStyle(.yellow)
                        }
                        if model.loaded {
                            Text("loaded")
                                .font(.caption)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.green.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        if model.id == controller.selectedModel {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(.blue)
                                .help("Active in Chat")
                        }
                        // Load only chat-capable models, but a loaded model is always
                        // unloadable so its memory can be freed regardless of kind.
                        if controller.busyModel == model.id {
                            ProgressView().controlSize(.small)
                        } else {
                            Button(model.loaded ? "Unload" : "Load") {
                                Task {
                                    if model.loaded {
                                        await controller.unload(model.id)
                                    } else {
                                        await controller.load(model.id)
                                    }
                                }
                            }
                            .disabled(!isChatLoadable(model.type) && !model.loaded)
                        }
                        // "Default" persists the model the Chat tab starts on, and switches
                        // the current session to it too. The reverse never happens: the Chat
                        // picker only changes the session, never the persisted default. The
                        // default now lives on the backend so Telegram uses it as well.
                        Button("Default") {
                            Task { await controller.setDefaultModel(model.id) }
                        }
                        .disabled(!isChatLoadable(model.type) || model.id == controller.defaultModel)
                        // Delete from disk (local models only; cached entries are shared).
                        Button(role: .destructive) { pendingDelete = model } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless).foregroundStyle(.red)
                        .disabled(model.source == "hf_cache")
                        .help(model.source == "hf_cache"
                            ? "Cached models are shared — remove with hf cache tools"
                            : "Delete this model from disk")
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .confirmationDialog(
            "Delete \(pendingDelete?.id ?? "")?",
            isPresented: Binding(get: { pendingDelete != nil },
                                 set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { model in
            Button("Delete from disk", role: .destructive) {
                Task { await controller.delete(model.id) }
            }
            Button("Cancel", role: .cancel) {}
        } message: { model in
            Text("Removes \(model.id) and its files from your model directory. "
                + "This can't be undone.")
        }
    }

    /// Chat works for text LLMs (mlx-lm) and vision-language / omni models (mlx-vlm).
    /// Embedding / text-encoder models aren't generative, so Load/Use are disabled for
    /// them. Unknown types (e.g. an omlx model_type) are allowed — the backend stays the
    /// source of truth and reports any real error.
    private func isChatLoadable(_ type: String?) -> Bool {
        guard let type else { return true }
        return type != "embedding"
    }

    /// Kind + on-disk size, e.g. "llm · 31.7 GB". Provenance is appended only for cached
    /// models ("HuggingFace cache"), where it explains why a model the user didn't place
    /// in their dir shows up; "models directory" was noise on every local row.
    private func subtitle(_ model: ModelDTO) -> String {
        var parts: [String] = []
        if let type = model.type {
            parts.append(isChatLoadable(type) ? type : "\(type) · not a chat model")
        }
        if let bytes = model.sizeBytes, bytes > 0 {
            parts.append(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
        }
        if model.source == "hf_cache" { parts.append("HuggingFace cache") }
        return parts.joined(separator: " · ")
    }
}

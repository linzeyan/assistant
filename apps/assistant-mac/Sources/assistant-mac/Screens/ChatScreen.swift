import AVKit
import AppKit
import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String  // "user" | "assistant" | "tool" | "media" | "diff"
    var text: String
    var mediaKind: String? = nil  // "image" | "video"
    var mediaPath: String? = nil
    var diff: String? = nil  // role "diff": the unified diff body, shown collapsible
}

/// A tool call paused awaiting the user's Approve/Deny.
struct PendingApproval: Identifiable {
    let id = UUID()
    let token: String
    let name: String
}

struct ChatScreen: View {
    @EnvironmentObject var controller: BackendController
    @EnvironmentObject var chat: ChatModel
    @State private var showSessions = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            if let pending = chat.pendingApproval {
                Divider()
                approvalBar(pending)
            }
            Divider()
            composer
        }
    }

    private var header: some View {
        HStack {
            Picker("Model", selection: Binding(
                get: { controller.selectedModel ?? "" },
                set: { controller.selectedModel = $0 }
            )) {
                if controller.models.isEmpty {
                    Text("no models").tag("")
                }
                ForEach(controller.models) { model in
                    Text(model.id).tag(model.id)
                }
            }
            .frame(maxWidth: 320)

            Spacer()

            if let ctx = chat.contextTokens {
                Text(Self.formatContext(ctx))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .help("Estimated context sent to the model last turn (tokens). "
                        + "Long sessions auto-compact before they exceed the model's window.")
                    .accessibilityLabel("Context \(ctx) tokens")
            }

            Button { showSessions.toggle() } label: { Image(systemName: "clock.arrow.circlepath") }
                .help("Conversations")
                .popover(isPresented: $showSessions, arrowEdge: .bottom) { sessionList }
            Button { chat.newChat() } label: { Image(systemName: "square.and.pencil") }
                .help("New chat")
                .disabled(chat.messages.isEmpty && chat.currentSessionId == nil)

            if !controller.modelBackendReady {
                Label(controller.reachable ? "no model backend" : "backend offline",
                      systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .font(.caption)
            }
        }
        .padding(8)
    }

    /// Saved-conversation list shown from the header. Loads on appearance; tap to resume,
    /// trash to delete, "New" to start fresh. Sessions persist on the backend (S1).
    private var sessionList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Conversations").font(.headline)
                Spacer()
                Button { chat.newChat(); showSessions = false } label: {
                    Label("New", systemImage: "square.and.pencil")
                }
                .buttonStyle(.borderless)
            }
            .padding(10)
            Divider()
            if chat.sessions.isEmpty {
                Text("No saved conversations yet.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            } else {
                List {
                    ForEach(chat.sessions) { session in
                        Button {
                            Task { await chat.openSession(session.id, client: controller.client) }
                            showSessions = false
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(session.title).lineLimit(1)
                                    Text(session.model ?? "—")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if session.id == chat.currentSessionId {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.blue)
                                }
                                Button(role: .destructive) {
                                    Task { await chat.deleteSession(session.id, client: controller.client) }
                                } label: { Image(systemName: "trash") }
                                    .buttonStyle(.borderless).foregroundStyle(.red)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .listStyle(.plain)
                .frame(height: 320)
            }
        }
        .frame(width: 340)
        .task { await chat.loadSessions(client: controller.client) }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(chat.messages) { message in
                        MessageRow(
                            message: message,
                            isStreaming: chat.streaming && message.id == chat.messages.last?.id
                        )
                    }
                }
                .padding()
            }
            .onChange(of: chat.messages.count) { _, _ in
                if let last = chat.messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private func approvalBar(_ pending: PendingApproval) -> some View {
        HStack {
            Image(systemName: "lock.shield")
            Text("Approve **\(pending.name)**?")
            Spacer()
            Button("Deny") { chat.respond(pending, false, client: controller.client) }
            Button("Approve") { chat.respond(pending, true, client: controller.client) }
                .keyboardShortcut(.defaultAction)
        }
        .padding(8)
        .background(Color.orange.opacity(0.15))
    }

    private var composer: some View {
        HStack(alignment: .bottom) {
            TextField("Message…", text: $chat.input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
                .onSubmit(send)
                .disabled(chat.streaming)
            if chat.streaming {
                Button(action: { chat.stop() }) {
                    Image(systemName: "stop.fill")
                }
                .keyboardShortcut(".", modifiers: .command)
                .help("Stop generating")
            } else {
                Button(action: send) {
                    Image(systemName: "paperplane.fill")
                }
                .keyboardShortcut(.return, modifiers: [])
                .disabled(!canSend)
            }
        }
        .padding(8)
    }

    private var canSend: Bool {
        !chat.streaming
            && controller.selectedModel != nil
            && !chat.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        guard let model = controller.selectedModel else { return }
        chat.send(model: model, client: controller.client) {
            await controller.refresh()
        }
    }

    /// Compact token count for the header badge: "~1.2k ctx" past a thousand, else "~840 ctx".
    static func formatContext(_ n: Int) -> String {
        n >= 1000 ? String(format: "~%.1fk ctx", Double(n) / 1000) : "~\(n) ctx"
    }
}

private struct MessageRow: View {
    let message: ChatMessage
    var isStreaming: Bool = false

    var body: some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 60) }
            content
            if message.role != "user" { Spacer(minLength: 60) }
        }
    }

    @ViewBuilder private var content: some View {
        if message.role == "media", let kind = message.mediaKind, let path = message.mediaPath {
            MediaView(kind: kind, path: path)
        } else if message.role == "diff" {
            DiffBubble(summary: message.text, diff: message.diff ?? "", background: background)
        } else if message.role == "assistant" {
            AssistantBubble(text: message.text, background: background, isStreaming: isStreaming)
        } else {
            Text(message.text.isEmpty ? "…" : message.text)
                .textSelection(.enabled)
                .padding(10)
                .background(background)
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var background: Color {
        switch message.role {
        case "user": return Color.accentColor.opacity(0.18)
        case "tool", "diff": return Color.secondary.opacity(0.12)
        default: return Color.gray.opacity(0.12)
        }
    }
}

/// A turn's file changes: a summary line that expands to the unified diff (monospaced,
/// scrollable, selectable). Collapsible so a large diff doesn't dominate the transcript.
private struct DiffBubble: View {
    let summary: String
    let diff: String
    let background: Color
    @State private var expanded = true

    var body: some View {
        Group {
            if diff.isEmpty {
                Text(summary)
            } else {
                DisclosureGroup(isExpanded: $expanded) {
                    ScrollView([.horizontal, .vertical]) {
                        Text(diff)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 4)
                    }
                    .frame(maxHeight: 320)
                } label: {
                    Text(summary).font(.callout)
                }
            }
        }
        .padding(10)
        .background(background)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// One piece of an assistant message: the visible answer (`prose`) or a collapsible
/// internal block — model reasoning (`think`) or tool-call markup (`toolCall`) — that
/// shouldn't pollute the answer or be copied.
/// Internal (not private) + Equatable so the parser is unit-testable.
enum MessageSegment: Equatable {
    case prose(String)
    case think(String)
    case toolCall(String)
}

/// Split assistant text into prose and the collapsible `<think>` / `<tool_call>` blocks.
/// An unterminated block (still streaming, or a truncated generation) collapses its
/// remainder rather than leaking raw markup into the answer.
func parseSegments(_ text: String) -> [MessageSegment] {
    let kinds: [(open: String, close: String, make: (String) -> MessageSegment)] = [
        ("<think>", "</think>", MessageSegment.think),
        ("<tool_call>", "</tool_call>", MessageSegment.toolCall),
    ]
    var out: [MessageSegment] = []
    var rest = text[...]
    // Some reasoning models (Qwen3.x) close their reasoning with </think> but never emit an
    // opening <think> — the chat template injects the opener into the generation prompt, so
    // the model only streams "<reasoning></think><answer>". Treat a </think> that precedes
    // any <think> as an implicit leading think block; otherwise its raw markup leaks into the
    // visible answer (the orphaned-</think> bug).
    if let close = rest.range(of: "</think>") {
        let open = rest.range(of: "<think>")
        if open == nil || close.lowerBound < open!.lowerBound {
            out.append(.think(String(rest[rest.startIndex..<close.lowerBound])))
            rest = rest[close.upperBound...]
        }
    }
    while !rest.isEmpty {
        var hit: (start: Substring.Index,
                  kind: (open: String, close: String, make: (String) -> MessageSegment))?
        for kind in kinds {
            if let r = rest.range(of: kind.open), hit == nil || r.lowerBound < hit!.start {
                hit = (r.lowerBound, kind)
            }
        }
        guard let hit else {
            out.append(.prose(String(rest)))
            break
        }
        if hit.start > rest.startIndex {
            out.append(.prose(String(rest[rest.startIndex..<hit.start])))
        }
        let afterOpen = rest.index(hit.start, offsetBy: hit.kind.open.count)
        let body = rest[afterOpen...]
        if let close = body.range(of: hit.kind.close) {
            out.append(hit.kind.make(String(body[body.startIndex..<close.lowerBound])))
            rest = body[close.upperBound...]
        } else {
            out.append(hit.kind.make(String(body)))
            break
        }
    }
    return out
}

/// Assistant bubble: collapsible think/tool blocks + prose, with a copy button that
/// copies only the prose answer (never reasoning or tool markup).
private struct AssistantBubble: View {
    let text: String
    let background: Color
    var isStreaming: Bool = false
    @State private var copied = false

    private var segments: [MessageSegment] { parseSegments(text) }

    private var proseText: String {
        segments
            .compactMap { if case .prose(let p) = $0 { return p } else { return nil } }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let p):
                    let trimmed = p.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        Text(trimmed).textSelection(.enabled)
                    }
                case .think(let inner):
                    CollapsibleBlock(title: "Thinking", systemImage: "brain", text: inner)
                case .toolCall(let inner):
                    CollapsibleBlock(title: "Tool call",
                                     systemImage: "wrench.and.screwdriver", text: inner)
                }
            }
            // Until prose streams in, show a working indicator (the model may be loading
            // or emitting a hidden <think> block) so the turn never looks frozen.
            if isStreaming && proseText.isEmpty {
                TypingIndicator()
            } else if segments.isEmpty {
                Text("…").foregroundStyle(.secondary)
            }
            if !proseText.isEmpty {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(proseText, forType: .string)
                    copied = true
                    Task {
                        try? await Task.sleep(nanoseconds: 1_200_000_000)
                        copied = false
                    }
                } label: {
                    Label(copied ? "Copied!" : "Copy",
                          systemImage: copied ? "checkmark.circle.fill" : "doc.on.doc")
                        .labelStyle(.iconOnly)
                        .font(.callout)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(copied ? Color.green : Color.secondary)
                .help(copied ? "Copied!" : "Copy")
            }
        }
        .padding(10)
        .background(background)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// A `<think>` / `<tool_call>` block, collapsed by default; tap to expand.
private struct CollapsibleBlock: View {
    let title: String
    let systemImage: String
    let text: String
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text.trimmingCharacters(in: .whitespacesAndNewlines))
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label(title, systemImage: systemImage)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

/// Animated "the model is working" indicator shown before any prose has streamed in.
private struct TypingIndicator: View {
    @State private var animating = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .frame(width: 6, height: 6)
                    .opacity(animating ? 1 : 0.25)
                    .animation(
                        .easeInOut(duration: 0.6).repeatForever().delay(Double(index) * 0.2),
                        value: animating
                    )
            }
        }
        .foregroundStyle(.secondary)
        .padding(.vertical, 2)
        .onAppear { animating = true }
        .accessibilityLabel("Generating")
    }
}

/// Inline preview of a generated image or video. Files are local (same machine as
/// the backend), so they're loaded straight off disk.
private struct MediaView: View {
    let kind: String
    let path: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if kind == "image", let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 360, maxHeight: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else if kind == "video" {
                VideoPlayer(player: AVPlayer(url: URL(fileURLWithPath: path)))
                    .frame(width: 360, height: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                Label("media unavailable", systemImage: "questionmark.square.dashed")
                    .foregroundStyle(.secondary)
            }
            Text(path)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }
}

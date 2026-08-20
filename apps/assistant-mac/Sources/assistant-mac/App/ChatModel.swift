import Foundation

/// Chat state lives here (not in `ChatScreen`'s `@State`) so it survives navigating
/// away to another tab and back — a `View`'s `@State` is torn down when the view leaves
/// the hierarchy, which previously wiped the transcript. Hosted once at the app root.
@MainActor
final class ChatModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var input: String = ""
    @Published var streaming = false
    @Published var pendingApproval: PendingApproval?

    /// This conversation's reasoning overrides (the Chat header menu), sent with every turn and
    /// beating the model's saved defaults. nil = use those defaults. Conversation-scoped rather
    /// than global: "think hard about this one" is a property of the task at hand, not of the
    /// model — so it resets whenever the transcript does, and isn't persisted with the session.
    @Published var thinkingOverride: Bool?
    @Published var effortOverride: String?

    @Published private(set) var currentSessionId: String?
    @Published var sessions: [SessionSummaryDTO] = []
    /// Estimated context size (tokens) of the last completed turn, from the `done` event.
    /// Shown in the Chat header so the user can watch the window fill before compaction.
    @Published private(set) var contextTokens: Int?
    private var streamTask: Task<Void, Never>?
    /// Index of this turn's plan bubble, so repeated `plan` events update it in place rather
    /// than appending a new checklist each time. Reset at the start of every turn (SA.3).
    private var planBubbleIndex: Int?

    func send(model: String, client: AssistantClient,
              onFinish: @escaping @MainActor () async -> Void = {}) {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !streaming else { return }
        input = ""
        messages.append(ChatMessage(role: "user", text: text))
        streaming = true
        streamTask = Task { @MainActor in
            await stream(text: text, model: model, client: client)
            streaming = false
            streamTask = nil
            // A chat turn loads its model on demand in the backend pool; let the caller
            // refresh so the Models tab reflects the now-loaded state (and offers Unload)
            // instead of showing a stale "Load".
            // Only on a turn that actually finished: Stop cancels this task, and running
            // onFinish in a cancelled context fires client calls that fail with
            // URLError.cancelled (-999) — which refresh() would misread as "backend offline".
            if !Task.isCancelled {
                await onFinish()
            }
        }
    }

    /// Cancel an in-flight turn. The streamed token task is cancelled and the SSE
    /// connection is torn down via the stream's `onTermination`.
    func stop() {
        guard streaming else { return }
        streamTask?.cancel()
        streamTask = nil
        streaming = false
        pendingApproval = nil
        messages.append(ChatMessage(role: "tool", text: "⏹ stopped"))
    }

    func respond(_ pending: PendingApproval, _ decision: Bool, client: AssistantClient) {
        pendingApproval = nil
        Task { try? await client.approveTool(token: pending.token, decision: decision) }
    }

    func clear() {
        stop()
        messages.removeAll()
        currentSessionId = nil
        contextTokens = nil
        thinkingOverride = nil
        effortOverride = nil
    }

    /// Start a fresh conversation; the backend assigns a new session id on the next turn.
    func newChat() { clear() }

    func loadSessions(client: AssistantClient) async {
        sessions = ((try? await client.listSessions())?.sessions) ?? []
    }

    /// Open a saved conversation: replace the transcript with its persisted messages and
    /// continue it (the backend appends to the same session file).
    func openSession(_ id: String, client: AssistantClient) async {
        guard let detail = try? await client.sessionDetail(id) else { return }
        stop()
        currentSessionId = detail.id
        contextTokens = nil  // unknown until this conversation's next turn reports usage
        // The overrides aren't persisted with the session, so a resumed conversation starts
        // from the model's defaults rather than inheriting the previous one's.
        thinkingOverride = nil
        effortOverride = nil
        messages = detail.messages.compactMap { m in
            guard m.role == "user" || m.role == "assistant" else { return nil }
            return ChatMessage(role: m.role, text: m.content ?? "")
        }
    }

    func deleteSession(_ id: String, client: AssistantClient) async {
        try? await client.deleteSession(id)
        if currentSessionId == id { newChat() }
        await loadSessions(client: client)
    }

    private func stream(text: String, model: String, client: AssistantClient) async {
        var currentAssistant: Int? = nil
        planBubbleIndex = nil  // a fresh turn starts a fresh plan bubble
        do {
            for try await event in client.chat(
                message: text, model: model, sessionId: currentSessionId,
                thinking: thinkingOverride, reasoningEffort: effortOverride
            ) {
                try Task.checkCancellation()
                reduce(event, currentAssistant: &currentAssistant)
            }
        } catch is CancellationError {
            // User pressed Stop — the transcript already shows the stop marker.
        } catch {
            messages.append(ChatMessage(role: "tool", text: "⚠️ \(error)"))
        }
        pendingApproval = nil  // turn ended; drop any stale prompt
    }

    /// Apply one streamed SSE event to the transcript. Factored out of `stream(...)` so
    /// the reducer is unit-testable without a live backend. `currentAssistant` tracks the
    /// index of the in-progress assistant bubble across consecutive `assistant_delta`s.
    func reduce(_ event: ChatEvent, currentAssistant: inout Int?) {
        switch event.type {
        case "session":
            currentSessionId = event.sessionId
        case "assistant_delta":
            if currentAssistant == nil {
                messages.append(ChatMessage(role: "assistant", text: ""))
                currentAssistant = messages.count - 1
            }
            if let chunk = event.content, let idx = currentAssistant {
                messages[idx].text += chunk
            }
        case "tool_call":
            currentAssistant = nil
            if let name = event.name {
                messages.append(ChatMessage(role: "tool", text: "⚙️ \(name)…"))
            }
        case "approval_request":
            currentAssistant = nil
            if let token = event.token, let name = event.name {
                pendingApproval = PendingApproval(token: token, name: name)
            }
        case "tool_result":
            currentAssistant = nil
            if event.ok == true, let path = event.content {
                switch event.name {
                case "generate_image":
                    messages.append(ChatMessage(role: "media", text: "", mediaKind: "image", mediaPath: path))
                case "generate_video":
                    messages.append(ChatMessage(role: "media", text: "", mediaKind: "video", mediaPath: path))
                default:
                    break
                }
            }
        case "turn_diff":
            currentAssistant = nil
            messages.append(ChatMessage(
                role: "diff",
                text: "✏️ \(event.summary ?? "files changed")",
                diff: event.diff))
        case "plan":
            // Update this turn's checklist in place (one bubble, ticked off live) rather than
            // appending a new one per update_plan call.
            currentAssistant = nil
            let steps = event.steps ?? []
            if let idx = planBubbleIndex, messages.indices.contains(idx) {
                messages[idx].planSteps = steps
            } else {
                messages.append(ChatMessage(role: "plan", text: "", planSteps: steps))
                planBubbleIndex = messages.count - 1
            }
        case "error":
            messages.append(ChatMessage(role: "tool", text: "⚠️ \(event.detail ?? "error")"))
        case "done":
            currentAssistant = nil
            if let ctx = event.usage?.contextTokens { contextTokens = ctx }
        default:
            break
        }
    }
}

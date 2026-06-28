import Foundation
import Testing

@testable import assistant_mac

/// The SSE reducer turns the backend's event stream into the visible transcript. These
/// pin the bubble-grouping rule (consecutive deltas share one assistant bubble; a
/// tool_call breaks the run) and the per-event-kind side effects, all without a backend.
@MainActor
struct ChatReducerTests {
    private func event(_ json: String) throws -> ChatEvent {
        try JSONDecoder().decode(ChatEvent.self, from: Data(json.utf8))
    }

    @Test func assistantDeltasAccumulateIntoOneBubble() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(try event(#"{"type":"assistant_delta","content":"Hel"}"#), currentAssistant: &current)
        model.reduce(try event(#"{"type":"assistant_delta","content":"lo"}"#), currentAssistant: &current)
        #expect(model.messages.count == 1)
        #expect(model.messages[0].role == "assistant")
        #expect(model.messages[0].text == "Hello")
    }

    @Test func sessionEventSetsSessionIdWithoutMessage() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(try event(#"{"type":"session","session_id":"sess-1"}"#), currentAssistant: &current)
        #expect(model.currentSessionId == "sess-1")
        #expect(model.messages.isEmpty)
    }

    @Test func toolCallBreaksAssistantRunAndAddsRow() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(try event(#"{"type":"assistant_delta","content":"hi"}"#), currentAssistant: &current)
        model.reduce(try event(#"{"type":"tool_call","name":"read_file"}"#), currentAssistant: &current)
        model.reduce(try event(#"{"type":"assistant_delta","content":"again"}"#), currentAssistant: &current)
        #expect(model.messages.count == 3)
        #expect(model.messages[1].role == "tool")
        #expect(model.messages[1].text.contains("read_file"))
        // A delta after the tool_call must start a NEW assistant bubble, not append to "hi".
        #expect(model.messages[2].role == "assistant")
        #expect(model.messages[2].text == "again")
    }

    @Test func approvalRequestSetsPending() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(
            try event(#"{"type":"approval_request","token":"tok","name":"bash"}"#),
            currentAssistant: &current)
        #expect(model.pendingApproval?.token == "tok")
        #expect(model.pendingApproval?.name == "bash")
    }

    @Test func toolResultImageBecomesMediaRow() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(
            try event(#"{"type":"tool_result","name":"generate_image","ok":true,"content":"/tmp/a.png"}"#),
            currentAssistant: &current)
        #expect(model.messages.count == 1)
        #expect(model.messages[0].role == "media")
        #expect(model.messages[0].mediaKind == "image")
        #expect(model.messages[0].mediaPath == "/tmp/a.png")
    }

    @Test func toolResultForNonMediaToolIsIgnored() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(
            try event(#"{"type":"tool_result","name":"read_file","ok":true,"content":"file body"}"#),
            currentAssistant: &current)
        #expect(model.messages.isEmpty)
    }

    @Test func turnDiffBecomesDiffRow() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(
            try event(#"{"type":"turn_diff","summary":"1 file changed (+1/-0)","diff":"+hi\n"}"#),
            currentAssistant: &current)
        #expect(model.messages.count == 1)
        #expect(model.messages[0].role == "diff")
        #expect(model.messages[0].text.contains("1 file changed"))
        #expect(model.messages[0].diff == "+hi\n")
    }

    @Test func errorEventAddsToolRowWithDetail() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(try event(#"{"type":"error","detail":"boom"}"#), currentAssistant: &current)
        #expect(model.messages.count == 1)
        #expect(model.messages[0].text.contains("boom"))
    }

    @Test func doneEventRecordsContextTokens() throws {
        let model = ChatModel()
        var current: Int?
        model.reduce(
            try event(#"{"type":"done","usage":{"context_tokens":1234,"output_tokens":56}}"#),
            currentAssistant: &current)
        #expect(model.contextTokens == 1234)
        #expect(model.messages.isEmpty)  // done is metadata-only, adds no bubble
    }

    @Test func contextBadgeFormatsCompactly() {
        #expect(ChatScreen.formatContext(840) == "~840 ctx")
        #expect(ChatScreen.formatContext(1234) == "~1.2k ctx")
    }
}

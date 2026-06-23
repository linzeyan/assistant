import Testing

@testable import assistant_mac

/// `parseSegments` decides what the user sees vs. what stays collapsed (reasoning /
/// tool markup) and what the Copy button copies. A regression here leaks raw `<think>`
/// markup into answers, so the boundary cases are pinned down.
struct ChatSegmentTests {
    @Test func plainProse() {
        #expect(parseSegments("hello world") == [.prose("hello world")])
    }

    @Test func emptyInputYieldsNoSegments() {
        #expect(parseSegments("") == [])
    }

    @Test func thinkBlockExtracted() {
        #expect(
            parseSegments("before<think>reasoning</think>after")
                == [.prose("before"), .think("reasoning"), .prose("after")])
    }

    @Test func toolCallBlockExtracted() {
        #expect(
            parseSegments(#"<tool_call>{"name":"x"}</tool_call>done"#)
                == [.toolCall(#"{"name":"x"}"#), .prose("done")])
    }

    @Test func unterminatedThinkCollapsesRemainder() {
        // A still-streaming / truncated block keeps its remainder collapsed instead of
        // leaking raw markup into the answer.
        #expect(
            parseSegments("answer<think>partial reasoning still going")
                == [.prose("answer"), .think("partial reasoning still going")])
    }

    @Test func multipleInterleavedBlocks() {
        #expect(
            parseSegments("a<think>t1</think>b<tool_call>c1</tool_call>c")
                == [.prose("a"), .think("t1"), .prose("b"), .toolCall("c1"), .prose("c")])
    }

    @Test func earliestOpenWinsAcrossKinds() {
        // tool_call opens before think → tool_call segment is emitted first.
        #expect(
            parseSegments("<tool_call>tc</tool_call><think>th</think>")
                == [.toolCall("tc"), .think("th")])
    }
}

import Testing

@testable import assistant_mac

/// `parseMarkdownBlocks` decides how assistant prose is laid out in the chat bubble.
/// The cases pinned here are the ones that used to read terribly as plain text — raw
/// code fences, headings, lists — plus the streaming edge (an unterminated fence).
struct MarkdownBlockTests {
    @Test func plainParagraph() {
        #expect(parseMarkdownBlocks("hello world") == [.paragraph("hello world")])
    }

    @Test func blankLineSplitsParagraphs() {
        #expect(
            parseMarkdownBlocks("one\n\ntwo")
                == [.paragraph("one"), .paragraph("two")])
    }

    @Test func softWrappedLinesStayOneParagraph() {
        #expect(parseMarkdownBlocks("one\ntwo") == [.paragraph("one\ntwo")])
    }

    @Test func headings() {
        #expect(
            parseMarkdownBlocks("# Title\nbody\n### Sub")
                == [
                    .heading(level: 1, text: "Title"),
                    .paragraph("body"),
                    .heading(level: 3, text: "Sub"),
                ])
    }

    @Test func hashesWithoutSpaceAreProse() {
        // "#hashtag" and a 7-# line are not headings.
        #expect(parseMarkdownBlocks("#hashtag") == [.paragraph("#hashtag")])
        #expect(parseMarkdownBlocks("####### x") == [.paragraph("####### x")])
    }

    @Test func fencedCodeBlock() {
        let text = "before\n```swift\nlet x = 1\n\nprint(x)\n```\nafter"
        #expect(
            parseMarkdownBlocks(text)
                == [
                    .paragraph("before"),
                    .code("let x = 1\n\nprint(x)"),
                    .paragraph("after"),
                ])
    }

    @Test func unterminatedFenceIsStillCode() {
        // Mid-stream: the fence hasn't closed yet — its content must render as code,
        // never leak backticks into prose.
        #expect(
            parseMarkdownBlocks("intro\n```\npartial")
                == [.paragraph("intro"), .code("partial")])
    }

    @Test func listMarkersInsideCodeAreNotItems() {
        #expect(
            parseMarkdownBlocks("```\n- not a list\n# not a heading\n```")
                == [.code("- not a list\n# not a heading")])
    }

    @Test func bulletAndOrderedLists() {
        #expect(
            parseMarkdownBlocks("- a\n* b\n1. c\n2) d")
                == [
                    .listItem(indent: 0, ordinal: nil, text: "a"),
                    .listItem(indent: 0, ordinal: nil, text: "b"),
                    .listItem(indent: 0, ordinal: "1", text: "c"),
                    .listItem(indent: 0, ordinal: "2", text: "d"),
                ])
    }

    @Test func nestedListIndentation() {
        #expect(
            parseMarkdownBlocks("- top\n  - nested")
                == [
                    .listItem(indent: 0, ordinal: nil, text: "top"),
                    .listItem(indent: 1, ordinal: nil, text: "nested"),
                ])
    }

    @Test func numberedProseIsNotAList() {
        // A year at line start must not become an ordered item.
        #expect(parseMarkdownBlocks("2026 was fine") == [.paragraph("2026 was fine")])
    }

    @Test func pipeTableGroupsConsecutiveLines() {
        #expect(
            parseMarkdownBlocks("| a | b |\n|---|---|\n| 1 | 2 |\ndone")
                == [
                    .table(["| a | b |", "|---|---|", "| 1 | 2 |"]),
                    .paragraph("done"),
                ])
    }

    @Test func dashLineAloneIsNotAListItem() {
        // "- " needs content; a bare "-" stays prose (e.g. a minus sign line).
        #expect(parseMarkdownBlocks("-") == [.paragraph("-")])
    }
}

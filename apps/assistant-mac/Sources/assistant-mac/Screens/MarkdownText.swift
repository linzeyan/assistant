import SwiftUI

/// One block of a markdown document. The chat renders assistant prose as markdown
/// (models emit it whether we render it or not); a small block parser + SwiftUI's
/// inline `AttributedString(markdown:)` covers what chat replies actually contain —
/// paragraphs, headings, fenced code, lists, tables — without a rendering dependency.
/// Internal (not private) + Equatable so the parser is unit-testable, like `parseSegments`.
enum MarkdownBlock: Equatable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case code(String)
    /// One list item; `ordinal` is nil for a bullet, the printed number for an ordered item.
    case listItem(indent: Int, ordinal: String?, text: String)
    /// Consecutive pipe-table lines, rendered monospaced (真 table layout 不值得為聊天泡泡做).
    case table([String])
}

/// Split markdown text into blocks. Line-based and single-pass, so re-parsing on every
/// streamed token stays cheap. An unterminated code fence (still streaming) renders as
/// code rather than leaking backticks into the prose.
func parseMarkdownBlocks(_ text: String) -> [MarkdownBlock] {
    var blocks: [MarkdownBlock] = []
    var paragraph: [String] = []
    var codeLines: [String]? = nil
    var tableLines: [String] = []

    func flushParagraph() {
        if !paragraph.isEmpty {
            blocks.append(.paragraph(paragraph.joined(separator: "\n")))
            paragraph = []
        }
    }
    func flushTable() {
        if !tableLines.isEmpty {
            blocks.append(.table(tableLines))
            tableLines = []
        }
    }

    for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        if var lines = codeLines {  // inside a fence: only the closing fence ends it
            if trimmed.hasPrefix("```") {
                blocks.append(.code(lines.joined(separator: "\n")))
                codeLines = nil
            } else {
                lines.append(String(line))
                codeLines = lines
            }
            continue
        }
        if trimmed.hasPrefix("```") {
            flushParagraph(); flushTable()
            codeLines = []  // language tag on the opening fence is dropped
            continue
        }
        if trimmed.hasPrefix("|") {
            flushParagraph()
            tableLines.append(trimmed)
            continue
        }
        flushTable()
        if trimmed.isEmpty {
            flushParagraph()
            continue
        }
        // Heading: 1-6 #'s followed by a space.
        if trimmed.hasPrefix("#") {
            let hashes = trimmed.prefix(while: { $0 == "#" })
            let rest = trimmed.dropFirst(hashes.count)
            if hashes.count <= 6 && rest.first == " " {
                flushParagraph()
                blocks.append(.heading(level: hashes.count,
                                       text: rest.trimmingCharacters(in: .whitespaces)))
                continue
            }
        }
        // List items: "- x" / "* x" / "+ x" / "1. x" / "1) x", indented or not.
        let indent = line.prefix(while: { $0 == " " }).count / 2
        if let marker = trimmed.first, "-*+".contains(marker),
           trimmed.dropFirst().first == " " {
            flushParagraph()
            blocks.append(.listItem(indent: indent, ordinal: nil,
                                    text: String(trimmed.dropFirst(2))))
            continue
        }
        let digits = trimmed.prefix(while: \.isNumber)
        if !digits.isEmpty, digits.count <= 3 {
            let after = trimmed.dropFirst(digits.count)
            if (after.hasPrefix(". ") || after.hasPrefix(") ")) {
                flushParagraph()
                blocks.append(.listItem(indent: indent, ordinal: String(digits),
                                        text: String(after.dropFirst(2))))
                continue
            }
        }
        paragraph.append(String(line))
    }
    if let lines = codeLines {  // unterminated fence (mid-stream): treat as code
        blocks.append(.code(lines.joined(separator: "\n")))
    }
    flushTable()
    flushParagraph()
    return blocks
}

/// Inline markdown (bold/italic/`code`/links) for one block's text. Full syntax is NOT
/// interpreted here — block structure came from `parseMarkdownBlocks`; this pass must
/// keep whitespace/newlines as written, hence `.inlineOnlyPreservingWhitespace`.
private func inline(_ s: String) -> AttributedString {
    (try? AttributedString(
        markdown: s,
        options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    )) ?? AttributedString(s)
}

/// Renders assistant prose as readable markdown. Selection stays enabled per block; the
/// bubble's Copy button copies the raw text, so what's on the clipboard is still markdown.
struct MarkdownText: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(parseMarkdownBlocks(text).enumerated()), id: \.offset) { _, block in
                render(block)
            }
        }
    }

    @ViewBuilder private func render(_ block: MarkdownBlock) -> some View {
        switch block {
        case .paragraph(let s):
            Text(inline(s)).textSelection(.enabled)
        case .heading(let level, let s):
            Text(inline(s))
                .font(level == 1 ? .title3.weight(.bold)
                    : level == 2 ? .headline
                    : .subheadline.weight(.semibold))
                .padding(.top, 2)
                .textSelection(.enabled)
        case .code(let code):
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
            }
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        case .listItem(let indent, let ordinal, let s):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(ordinal.map { "\($0)." } ?? "•")
                    .font(.callout)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                Text(inline(s)).textSelection(.enabled)
            }
            .padding(.leading, CGFloat(indent) * 14)
        case .table(let lines):
            ScrollView(.horizontal) {
                Text(lines.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
            }
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

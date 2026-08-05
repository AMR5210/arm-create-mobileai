import SwiftUI

/// Renders a model response as formatted Markdown.
///
/// SwiftUI's `Text` accepts an `AttributedString` parsed from Markdown, but that
/// covers inline syntax only -- `**bold**` renders, while `### Heading` and list
/// markers stay as literal characters. Model output uses both, so blocks are split
/// out here and given their own font, with inline parsing applied within each.
struct MarkdownText: View {
    let markdown: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block.kind {
                case .heading(let level):
                    Text(inline(block.text))
                        .font(level <= 1 ? .headline : level == 2 ? .subheadline : .footnote)
                        .bold()
                case .bullet:
                    HStack(alignment: .top, spacing: 6) {
                        Text("•").font(.footnote)
                        Text(inline(block.text)).font(.footnote)
                    }
                case .paragraph:
                    Text(inline(block.text)).font(.footnote)
                }
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Block splitting

    private enum Kind {
        case heading(Int)
        case bullet
        case paragraph
    }

    private struct Block {
        var kind: Kind
        var text: String
    }

    private var blocks: [Block] {
        var out: [Block] = []
        for rawLine in markdown.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }

            if line.hasPrefix("#") {
                let hashes = line.prefix { $0 == "#" }.count
                let text = line.dropFirst(hashes).trimmingCharacters(in: .whitespaces)
                out.append(Block(kind: .heading(hashes), text: text))
            } else if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("+ ") {
                out.append(Block(kind: .bullet, text: String(line.dropFirst(2))))
            } else if let m = line.range(of: "^[0-9]+[.)] ", options: .regularExpression) {
                out.append(Block(kind: .bullet, text: String(line[m.upperBound...])))
            } else if case .paragraph = out.last?.kind ?? .heading(0), var last = out.last {
                // Consecutive plain lines belong to one paragraph.
                last.text += " " + line
                out[out.count - 1] = last
            } else {
                out.append(Block(kind: .paragraph, text: line))
            }
        }
        return out
    }

    /// Inline Markdown for one block, falling back to the raw text when the string
    /// does not parse -- truncated output can end mid-token and leave unbalanced
    /// emphasis markers.
    private func inline(_ s: String) -> AttributedString {
        (try? AttributedString(
            markdown: s,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(s)
    }
}

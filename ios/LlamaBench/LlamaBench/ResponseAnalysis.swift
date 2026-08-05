import Foundation

/// Classifies a generated response so the UI can label it rather than presenting
/// a degenerate loop or a mid-sentence cut as if it were an answer.
enum ResponseAnalysis {

    struct Flags {
        /// Generation stopped at the cap without the model emitting end-of-sequence.
        var truncated = false
        /// A short phrase repeats consecutively for a large share of the response.
        var repetition = false
        /// Length of the repeating unit in words, when `repetition` is set.
        var repeatUnitWords = 0
        /// How many consecutive times that unit repeats.
        var repeatCount = 0
    }

    /// Words, lowercased, with markdown emphasis and punctuation stripped.
    ///
    /// Normalising first lets a loop be recognised regardless of decoration: the
    /// observed cases include `**الملك**` separated by blank lines and `how to know`
    /// repeated inside running prose.
    static func normalizedWords(_ text: String) -> [String] {
        let stripped = text.replacingOccurrences(of: "[*_`#>]", with: " ",
                                                 options: .regularExpression)
        return stripped
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    /// Detects a phrase repeating consecutively.
    ///
    /// Scans unit lengths from 1 to 8 words and finds the longest run of identical
    /// consecutive blocks. A response is flagged when a unit repeats at least
    /// `minRepeats` times in a row **and** the repeated span covers at least
    /// `minCoverage` of the response.
    ///
    /// Both conditions are needed to separate a degenerate loop from legitimate
    /// repetition. An answer may restate a phrase two or three times, as QAT does
    /// on arithmetic prompts, while a collapsed model emits one unit for nearly the
    /// whole response. Thresholds of 4 repeats and 50% coverage sit between the two
    /// behaviours measured here.
    static func detectRepetition(_ text: String,
                                 minRepeats: Int = 4,
                                 minCoverage: Double = 0.5) -> (unit: Int, count: Int)? {
        let words = normalizedWords(text)
        guard words.count >= minRepeats else { return nil }

        var best: (unit: Int, count: Int, covered: Int)?
        for unit in 1...min(8, words.count / minRepeats) {
            var i = 0
            while i + unit <= words.count {
                let block = Array(words[i..<(i + unit)])
                var count = 1
                var j = i + unit
                while j + unit <= words.count, Array(words[j..<(j + unit)]) == block {
                    count += 1
                    j += unit
                }
                if count >= minRepeats {
                    let covered = count * unit
                    if best == nil || covered > best!.covered {
                        best = (unit, count, covered)
                    }
                }
                i += max(1, count > 1 ? count * unit : 1)
            }
        }

        guard let b = best,
              Double(b.covered) / Double(words.count) >= minCoverage else { return nil }
        return (b.unit, b.count)
    }

    static func flags(text: String, stopReason: LlamaRunner.StopReason) -> Flags {
        var f = Flags()
        f.truncated = stopReason == .tokenLimit
        if let r = detectRepetition(text) {
            f.repetition = true
            f.repeatUnitWords = r.unit
            f.repeatCount = r.count
        }
        return f
    }

    // MARK: - Cross-panel agreement

    /// Similarity between two responses, 0...1, over normalized words.
    ///
    /// Uses the length of the common prefix relative to the longer response, which
    /// suits this comparison: variants that agree do so from the first word, and a
    /// variant that diverges does so and does not recover.
    static func similarity(_ a: String, _ b: String) -> Double {
        let wa = normalizedWords(a), wb = normalizedWords(b)
        guard !wa.isEmpty, !wb.isEmpty else { return wa.isEmpty && wb.isEmpty ? 1 : 0 }
        var common = 0
        for (x, y) in zip(wa, wb) {
            if x != y { break }
            common += 1
        }
        return Double(common) / Double(max(wa.count, wb.count))
    }

    /// Tags of responses that agree with at least one other response.
    ///
    /// 0.9 admits differences in trailing punctuation or a final partial word while
    /// requiring the substance to match, so agreement between two variants is
    /// reported only when their answers are equivalent.
    static func matchingTags(_ responses: [String: String],
                             threshold: Double = 0.9) -> Set<String> {
        var matched = Set<String>()
        let keys = Array(responses.keys)
        for i in 0..<keys.count {
            for j in (i + 1)..<keys.count {
                guard let a = responses[keys[i]], let b = responses[keys[j]],
                      !normalizedWords(a).isEmpty else { continue }
                if similarity(a, b) >= threshold {
                    matched.insert(keys[i])
                    matched.insert(keys[j])
                }
            }
        }
        return matched
    }
}

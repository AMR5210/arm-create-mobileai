import Foundation

/// Instruction-following eval, mirroring `scripts/instruction_eval.py`.
///
/// Prompt format is worth stating precisely, since it is not evident from the
/// desktop script's flags. `ask_model()` passes `-no-cnv` alongside
/// `--chat-template-kwargs '{"enable_thinking": false}'`. Measured against
/// llama-cli at the checked-out revision, `-no-cnv` does not suppress the chat
/// template for a model that ships one: runs with and without it are identical,
/// while adding or removing the thinking kwarg changes the output. The desktop
/// eval therefore scores the question wrapped in Qwen3's chat template with
/// thinking disabled, and this implementation matches it.
///
/// Rendering that template for a single user turn with `add_generation_prompt`
/// and `enable_thinking=false` produces:
///
///     <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
///
/// It is applied literally here. The xcframework is built without llama.cpp's
/// `common` library, so the jinja templating path is unavailable in-process;
/// the rendered form is pinned instead, and `scripts/compare_generations.sh`
/// exercises the same format on the desktop side for cross-checking.
enum InstructionEval {

    struct Question: Decodable {
        let subject: String
        let question: String
        let choices: [String]
        let answer: Int
    }

    /// Forced-choice accuracy is the reported metric: the model's preferred
    /// answer among exactly A/B/C/D, read from the logits at the first generated
    /// position. Chance is 25%, every question yields a prediction, and the
    /// figure does not depend on output formatting or generation budget.
    ///
    /// The two generate-and-parse figures are retained as diagnostics. Parse rate
    /// (whether free generation yields a well-formed letter) reflects conformance
    /// to this prompt template as well as capability, and varies with the token
    /// budget for some variants. Free-generation accuracy sits at or below chance
    /// for every variant measured, including fp16
    /// (results/logs/instr_eval_diag_summary.json).
    struct Outcome {
        var forcedChoiceAccuracy: Double
        var perSubjectForcedChoiceAccuracy: [String: Double]
        var forcedChoiceCorrect: Int
        var forcedChoicePredictions: String        // one letter per question, in order
        var forcedChoiceHistogram: [String: Int]
        var total: Int

        // Diagnostics from the free-generation path.
        var parseRate: Double?
        var perSubjectParseRate: [String: Double]?
        var parsed: Int?
        var accuracyDiagnostic: Double?
        var perSubjectAccuracyDiagnostic: [String: Double]?
    }

    static let letters = Array("ABCD")

    /// Verbatim port of `build_prompt()`.
    static func buildPrompt(question: String, choices: [String]) -> String {
        var lines = [
            "Answer the following multiple-choice question with only the letter of the "
            + "correct choice.",
            "",
            "Question: \(question)",
        ]
        for (letter, choice) in zip(letters, choices) {
            lines.append("\(letter). \(choice)")
        }
        lines.append("Answer:")
        return lines.joined(separator: "\n")
    }

    /// Wraps a user turn in Qwen3's chat template with thinking disabled.
    static func applyChatTemplate(_ content: String) -> String {
        "<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// First standalone A-D letter, matching Python's `\b([ABCD])\b`.
    static func extractLetter(_ text: String) -> Character? {
        // The desktop script scans only what follows the last "Answer:", since
        // llama-cli echoes the prompt into stdout. Only generated text reaches
        // here, but the same rule is applied for equivalence when the model
        // emits its own "Answer:".
        let scanned = text.components(separatedBy: "Answer:").last ?? text
        let chars = Array(scanned)
        for (i, c) in chars.enumerated() where "ABCD".contains(c) {
            let prevOK = i == 0 || !(chars[i - 1].isLetter || chars[i - 1].isNumber || chars[i - 1] == "_")
            let nextIdx = i + 1
            let nextOK = nextIdx >= chars.count
                || !(chars[nextIdx].isLetter || chars[nextIdx].isNumber || chars[nextIdx] == "_")
            if prevOK && nextOK { return c }
        }
        return nil
    }

    static func loadQuestions(from url: URL) throws -> [Question] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        return try text.split(separator: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { try decoder.decode(Question.self, from: Data($0.utf8)) }
    }

    /// Forced-choice pass: one prompt evaluation per question, no generation.
    /// A single decode rather than N sampled tokens, so it is both faster than the
    /// generate-and-parse path and independent of output format.
    static func runForcedChoice(runner: LlamaRunner,
                                questions: [Question],
                                debugFirstN: Int = 0,
                                log: (@Sendable (String) -> Void)? = nil,
                                progress: (@Sendable (Int, Int) -> Void)? = nil) async throws -> Outcome {
        // Resolve the four answer labels to single token ids once.
        var ids: [Int32] = []
        for l in letters {
            guard let id = try await runner.singleTokenId(for: String(l)) else {
                throw NSError(domain: "LlamaBench", code: 5, userInfo: [
                    NSLocalizedDescriptionKey:
                        "answer label '\(l)' does not encode to a single token; "
                        + "forced-choice scoring cannot compare it"])
            }
            ids.append(id)
        }
        log?("      forced-choice token ids: "
             + zip(letters, ids).map { "\($0)=\($1)" }.joined(separator: " "))

        var correct = 0
        var preds = ""
        var hist: [String: Int] = [:]
        var subjectCorrect: [String: Int] = [:]
        var subjectTotal: [String: Int] = [:]

        for (i, q) in questions.enumerated() {
            let prompt = applyChatTemplate(buildPrompt(question: q.question, choices: q.choices))
            let (best, scores) = try await runner.argmaxOverCandidates(prompt: prompt, candidates: ids)
            let idx = ids.firstIndex(of: best) ?? 0
            let letter = letters[idx]
            let isCorrect = idx == q.answer

            if i < debugFirstN, let log {
                let detail = zip(letters, scores).map { String(format: "%@=%.3f", String($0), $1) }
                    .joined(separator: " ")
                log("      [debug q\(i)] logits \(detail) -> \(letter) "
                    + "(expected \(letters[q.answer]))")
            }

            preds.append(letter)
            hist[String(letter), default: 0] += 1
            correct += isCorrect ? 1 : 0
            subjectTotal[q.subject, default: 0] += 1
            subjectCorrect[q.subject, default: 0] += isCorrect ? 1 : 0
            progress?(i + 1, questions.count)
        }

        var perSubject: [String: Double] = [:]
        for (s, total) in subjectTotal {
            perSubject[s] = total > 0 ? Double(subjectCorrect[s] ?? 0) / Double(total) : 0
        }

        return Outcome(forcedChoiceAccuracy: Double(correct) / Double(max(questions.count, 1)),
                       perSubjectForcedChoiceAccuracy: perSubject,
                       forcedChoiceCorrect: correct,
                       forcedChoicePredictions: preds,
                       forcedChoiceHistogram: hist,
                       total: questions.count,
                       parseRate: nil, perSubjectParseRate: nil, parsed: nil,
                       accuracyDiagnostic: nil, perSubjectAccuracyDiagnostic: nil)
    }

    static func run(runner: LlamaRunner,
                    questions: [Question],
                    maxTokens: Int = 8,
                    debugFirstN: Int = 0,
                    log: (@Sendable (String) -> Void)? = nil,
                    progress: (@Sendable (Int, Int) -> Void)? = nil) async throws -> Outcome {
        var correct = 0
        var parsed = 0
        var subjectCorrect: [String: Int] = [:]
        var subjectParsed: [String: Int] = [:]
        var subjectTotal: [String: Int] = [:]

        for (i, q) in questions.enumerated() {
            let prompt = applyChatTemplate(buildPrompt(question: q.question, choices: q.choices))

            if i < debugFirstN, let log {
                let toks = try await runner.tokenize(prompt, addSpecial: false, parseSpecial: true)
                log("      [debug q\(i)] prompt \(toks.count) tokens; "
                    + "head=\(toks.prefix(4).map(String.init).joined(separator: ",")) "
                    + "tail=\(toks.suffix(6).map(String.init).joined(separator: ","))")
                log("      [debug q\(i)] prompt repr: \(prompt.debugDescription.prefix(120))…")
            }
            // addSpecial: false  — the template supplies its own control tokens.
            // parseSpecial: true — required so that <|im_start|> and similar are
            //                      matched as control tokens rather than as
            //                      literal text.
            let (text, _) = try await runner.generate(prompt: prompt,
                                                     maxTokens: maxTokens,
                                                     addSpecial: false,
                                                     parseSpecial: true)
            if i < debugFirstN, let log {
                log("      [debug q\(i)] generated: \(text.debugDescription)")
            }
            let letter = extractLetter(text)
            let predictedIdx = letter.flatMap { letters.firstIndex(of: $0) } ?? -1
            let isCorrect = predictedIdx == q.answer
            if letter != nil { parsed += 1 }
            correct += isCorrect ? 1 : 0
            subjectTotal[q.subject, default: 0] += 1
            subjectCorrect[q.subject, default: 0] += isCorrect ? 1 : 0
            subjectParsed[q.subject, default: 0] += letter != nil ? 1 : 0
            progress?(i + 1, questions.count)
        }

        var perSubjectParse: [String: Double] = [:]
        var perSubjectAcc: [String: Double] = [:]
        for (s, total) in subjectTotal {
            perSubjectParse[s] = total > 0 ? Double(subjectParsed[s] ?? 0) / Double(total) : 0
            perSubjectAcc[s] = total > 0 ? Double(subjectCorrect[s] ?? 0) / Double(total) : 0
        }

        let n = Double(max(questions.count, 1))
        return Outcome(forcedChoiceAccuracy: 0, perSubjectForcedChoiceAccuracy: [:],
                       forcedChoiceCorrect: 0, forcedChoicePredictions: "",
                       forcedChoiceHistogram: [:],
                       total: questions.count,
                       parseRate: Double(parsed) / n,
                       perSubjectParseRate: perSubjectParse,
                       parsed: parsed,
                       accuracyDiagnostic: Double(correct) / n,
                       perSubjectAccuracyDiagnostic: perSubjectAcc)
    }
}

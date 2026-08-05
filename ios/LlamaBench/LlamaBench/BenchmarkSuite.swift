import CryptoKit
import Foundation

/// Runs the five benchmark metrics for one model variant and emits a
/// `BenchmarkRecord`.
///
/// Metric definitions and their desktop counterparts:
///
///  1. Disk size            — file size of the .gguf (same as desktop).
///  2. Peak RAM             — see `MemorySampler`; cross-variant comparable.
///  3. Throughput           — pp-512 / tg-128, 5 reps (llama-bench defaults).
///  4. Perplexity           — llama-perplexity replication, see
///                            `LlamaRunner.perplexity`.
///  5. Instruction accuracy — see `InstructionEval`.
///
/// One model is resident at a time: the runner is released and the sampler
/// restarted per variant, so peak RAM is not inflated by a previous run.
enum BenchmarkSuite {

    struct Options {
        var backend: Backend = .cpuOnly
        var device: String = "unknown device"
        var perplexityChunkLimit: Int? = nil        // nil = full corpus
        var skipPerplexity = false
        var skipInstructionEval = false
        var skipThroughput = false
        var throughputReps = 5
        var instructionEvalLimit: Int? = nil        // nil = all questions
        var instructionEvalDebugFirstN = 0
        /// Generation budget per question for the diagnostic generate-and-parse
        /// path. 8 matches the desktop script; larger values distinguish a budget
        /// limit from an inability to answer.
        var instructionEvalMaxTokens = 8
        /// Takes every Nth question. instruction_eval.jsonl is grouped by subject
        /// (25 each), so a prefix sample covers a single subject while a stride
        /// spans all four.
        var instructionEvalStride = 1
        var checkpointEvery = 25
        var resumeFromCheckpoint = true
        var outputDirectory: URL = ResourceLocator.outputDirectory
        /// Overrides the output filename stem. A partial run (for example
        /// instruction-eval only) writes to its own file rather than overwriting
        /// a complete record; `scripts/merge_ios_record.py` folds it in.
        var outputBasename: String? = nil
    }

    /// Sidecar file holding resumable perplexity state, alongside the record it
    /// belongs to. Removed once the variant's perplexity completes.
    static func checkpointURL(tag: String, in dir: URL) -> URL {
        dir.appendingPathComponent("\(tag).ppl-checkpoint.json")
    }

    static func loadCheckpoint(tag: String, in dir: URL,
                               expectSHA: String?, expectCorpusTokens: Int,
                               expectNCtx: Int) -> PerplexityCheckpoint? {
        let url = checkpointURL(tag: tag, in: dir)
        guard let data = try? Data(contentsOf: url),
              let cp = try? JSONDecoder().decode(PerplexityCheckpoint.self, from: data)
        else { return nil }
        // Only resume when the checkpoint describes the same measurement.
        guard cp.corpusTokens == expectCorpusTokens, cp.nCtx == expectNCtx else { return nil }
        if let want = expectSHA, let got = cp.modelSHA256, want != got { return nil }
        guard cp.chunksDone > 0, cp.chunksDone < cp.totalChunks else { return nil }
        return cp
    }

    static func saveCheckpoint(_ cp: PerplexityCheckpoint, tag: String, in dir: URL) {
        var cp = cp
        cp.tag = tag
        cp.savedAt = BenchmarkRecord.timestamp()
        guard let data = try? JSONEncoder().encode(cp) else { return }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Write to a temp file then move, so an interruption mid-write cannot
        // leave a truncated checkpoint that would be rejected on resume.
        let final = checkpointURL(tag: tag, in: dir)
        let tmp = final.appendingPathExtension("tmp")
        do {
            try data.write(to: tmp)
            _ = try? FileManager.default.removeItem(at: final)
            try FileManager.default.moveItem(at: tmp, to: final)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
        }
    }

    static func clearCheckpoint(tag: String, in dir: URL) {
        try? FileManager.default.removeItem(at: checkpointURL(tag: tag, in: dir))
    }

    static func sha256(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try? handle.read(upToCount: 8 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func run(variant: ModelVariant,
                    options: Options,
                    log: @escaping @Sendable (String) -> Void) async throws -> BenchmarkRecord {

        guard let modelURL = ResourceLocator.locate(variant) else {
            throw NSError(domain: "LlamaBench", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "\(variant.fileName) not found. Searched:\n  "
                    + ResourceLocator.searchPaths.joined(separator: "\n  ")])
        }

        log("=== \(variant.displayName) (\(variant.tag)) ===")
        log("model: \(modelURL.lastPathComponent)")

        let diskBytes = ResourceLocator.fileSize(modelURL)
        log("[1/5] disk: \(String(format: "%.1f MB", Double(diskBytes) / 1e6))")

        let digest = sha256(of: modelURL)
        log("      sha256: \(digest?.prefix(16) ?? "n/a")…")

        let sampler = MemorySampler()
        sampler.start()

        let threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        let runner = try LlamaRunner.load(path: modelURL.path, backend: options.backend)
        let nCtx = await runner.contextSize
        log("      loaded; backend=\(options.backend.label) threads=\(threads) n_ctx=\(nCtx)")

        // --- 3. Throughput -------------------------------------------------
        var tp: ThroughputResult?
        if !options.skipThroughput {
            log("[2/5] throughput: pp512 / tg128 x \(options.throughputReps) reps")
            tp = try await runner.throughput(reps: options.throughputReps) { rep, total in
                log("      rep \(rep)/\(total)")
            }
            log(String(format: "      prompt %.2f tok/s   gen %.2f tok/s",
                       tp!.promptTokensPerSec, tp!.genTokensPerSec))
        } else {
            log("[2/5] throughput: skipped")
        }

        // --- 4. Perplexity -------------------------------------------------
        var ppl: PerplexityResult?
        if !options.skipPerplexity, let corpus = ResourceLocator.locateEval("wikitext2_test.txt") {
            log("[3/5] perplexity: wikitext2, n_ctx=512, non-overlapping chunks, scoring [256,511)")
            let text = try String(contentsOf: corpus, encoding: .utf8)
            var tokens = try await runner.tokenize(text, addSpecial: true)
            log("      corpus tokens: \(tokens.count)")
            if let limit = options.perplexityChunkLimit {
                tokens = Array(tokens.prefix(limit * 512))
                log("      limited to \(limit) chunks (partial corpus)")
            }

            let dir = options.outputDirectory
            var resume: PerplexityCheckpoint?
            if options.resumeFromCheckpoint {
                resume = loadCheckpoint(tag: variant.tag, in: dir, expectSHA: digest,
                                        expectCorpusTokens: tokens.count, expectNCtx: 512)
                if let r = resume {
                    log(String(format: "      resuming from checkpoint: %d/%d chunks done, running ppl %.4f",
                               r.chunksDone, r.totalChunks, r.runningPpl))
                }
            }

            let t0 = Date()
            let tagCopy = variant.tag
            let shaCopy = digest
            ppl = try await runner.perplexity(
                tokens: tokens,
                resumeFrom: resume,
                checkpointEvery: options.checkpointEvery,
                onCheckpoint: { cp in
                    var cp = cp
                    cp.modelSHA256 = shaCopy
                    saveCheckpoint(cp, tag: tagCopy, in: dir)
                },
                progress: { chunk, total, running in
                    if chunk % 50 == 0 || chunk == total {
                        log(String(format: "      chunk %d/%d  running ppl %.4f", chunk, total, running))
                    }
                })
            clearCheckpoint(tag: variant.tag, in: dir)
            log(String(format: "      perplexity %.4f over %d chunks / %d tokens (%.0fs)%@",
                       ppl!.ppl, ppl!.chunks, ppl!.tokensScored, Date().timeIntervalSince(t0),
                       ppl!.resumedFromChunk != nil ? " [resumed]" : ""))
        } else {
            log("[3/5] perplexity: skipped")
        }

        // --- 5. Instruction eval -------------------------------------------
        var instr: InstructionEval.Outcome?
        if !options.skipInstructionEval,
           let evalURL = ResourceLocator.locateEval("instruction_eval.jsonl") {
            var questions = try InstructionEval.loadQuestions(from: evalURL)
            if options.instructionEvalStride > 1 {
                let stride = options.instructionEvalStride
                questions = questions.enumerated().filter { $0.offset % stride == 0 }.map { $0.element }
                log("[4/5] instruction eval: stride \(stride) -> \(questions.count) questions "
                    + "spanning subjects")
            }
            if let limit = options.instructionEvalLimit, limit < questions.count {
                questions = Array(questions.prefix(limit))
                log("[4/5] instruction eval: SAMPLE of \(questions.count) questions (not the full set)")
            }
            log("[4/5] instruction eval: \(questions.count) questions, chat template, "
                + "thinking disabled, budget \(options.instructionEvalMaxTokens) tokens")
            instr = try await InstructionEval.runForcedChoice(
                runner: runner, questions: questions,
                debugFirstN: options.instructionEvalDebugFirstN,
                log: log,
                progress: { i, total in
                    if i % 25 == 0 || i == total { log("      \(i)/\(total)") }
                })
            log(String(format: "      forced-choice accuracy %.1f%%  (%d/%d correct, chance 25%%)",
                       instr!.forcedChoiceAccuracy * 100, instr!.forcedChoiceCorrect, instr!.total))
            log("      predictions: \(instr!.forcedChoicePredictions)")
            log("      histogram:   \(instr!.forcedChoiceHistogram.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
        } else {
            log("[4/5] instruction eval: skipped")
        }

        // --- 2. Peak RAM ----------------------------------------------------
        sampler.sample()
        let peak = sampler.peakBytes
        sampler.stop()
        log(String(format: "[5/5] peak RAM: %.1f MB (whole-process; cross-variant comparable)",
                   Double(peak) / 1e6))

        let kai = LlamaLog.shared.kleidiaiLines()

        return BenchmarkRecord(
            tag: variant.tag,
            device: options.device,
            model_path: modelURL.lastPathComponent,
            timestamp_utc: BenchmarkRecord.timestamp(),
            disk_bytes: diskBytes,
            peak_ram_bytes: peak,
            prompt_tokens_per_sec: tp?.promptTokensPerSec,
            gen_tokens_per_sec: tp?.genTokensPerSec,
            perplexity: ppl?.ppl,
            instruction_forced_choice_accuracy: instr?.forcedChoiceAccuracy,
            instruction_per_subject_forced_choice_accuracy: instr?.perSubjectForcedChoiceAccuracy,
            llama_bench_raw: nil,
            model_sha256: digest,
            backend: options.backend.label,
            harness: .init(
                app: "LlamaBench (ios/LlamaBench)",
                llama_cpp_build: kai.isEmpty ? "llama.xcframework" : "llama.xcframework (KleidiAI active)",
                arm_arch: "armv8.6-a+dotprod+i8mm+fp16",
                threads: threads,
                n_ctx: nCtx,
                perplexity_method:
                    "llama-perplexity replication: non-overlapping 512-token chunks, KV cleared per "
                    + "chunk, scoring positions [256,511) (n_ctx-first-1 = 255 per chunk), "
                    + "ppl = exp(nll/count)",
                perplexity_corpus: "eval/data/wikitext2_test.txt",
                perplexity_chunks: ppl?.chunks,
                perplexity_tokens_scored: ppl?.tokensScored,
                throughput_method: tp.map {
                    "pp\($0.nPrompt) / tg\($0.nGen), \($0.reps) reps, KV cleared between passes (llama-bench defaults)"
                } ?? "not measured in this run",
                throughput_prompt_samples: tp?.promptSamples,
                throughput_gen_samples: tp?.genSamples,
                instruction_eval_prompt_format:
                    "Qwen3 chat template, enable_thinking=false: "
                    + "<|im_start|>user\\n{build_prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n",
                instruction_eval_total: instr?.total,
                instruction_eval_method: instr == nil ? nil :
                    "forced choice: argmax over the A/B/C/D logits at the first generated position",
                instruction_eval_forced_choice_correct: instr?.forcedChoiceCorrect,
                instruction_eval_forced_choice_predictions: instr?.forcedChoicePredictions,
                instruction_eval_forced_choice_histogram: instr?.forcedChoiceHistogram,
                instruction_eval_parse_rate_diagnostic: instr?.parseRate,
                instruction_eval_parsed: instr?.parsed,
                instruction_eval_accuracy_diagnostic: instr?.accuracyDiagnostic,
                instruction_eval_per_subject_accuracy_diagnostic: instr?.perSubjectAccuracyDiagnostic,
                instruction_eval_metric_note: instr == nil ? nil :
                    "Reported metric is forced-choice accuracy over A/B/C/D (chance 25%), "
                    + "independent of output formatting and generation budget. The "
                    + "generate-and-parse figures alongside it are diagnostics.",
                peak_ram_scope:
                    "Whole-app resident_size via task_info/MACH_TASK_BASIC_INFO, sampled at 100ms. "
                    + "Comparable across the three variants measured here; NOT comparable to the "
                    + "desktop harness, which records ru_maxrss of a separate llama-bench child process."
            ))
    }

    /// Writes one record per variant into `directory`, named `<tag>.json` to
    /// match what `scripts/summarize_results.py` reads.
    static func write(_ record: BenchmarkRecord, to directory: URL,
                      basename: String? = nil) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let out = directory.appendingPathComponent("\(basename ?? record.tag).json")
        try record.encodedJSON().write(to: out)
        return out
    }
}

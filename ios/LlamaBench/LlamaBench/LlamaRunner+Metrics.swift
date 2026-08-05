import Accelerate
import Foundation
import llama

extension LlamaRunner {

    // MARK: - Perplexity

    /// Replicates llama.cpp's `llama-perplexity` measurement exactly, so the
    /// number is comparable to the figures recorded in `results/`.
    ///
    /// Method, from `tools/perplexity/perplexity.cpp` at the checked-out revision:
    ///
    ///  - Tokenize the whole corpus once with special tokens enabled.
    ///  - Split into **non-overlapping** chunks of `nCtx` tokens
    ///    (`start = i * n_ctx`); `n_chunk = n_tokens / n_ctx`.
    ///  - Clear the KV cache before each chunk, so chunks are independent.
    ///  - Score only the second half of each chunk: `first = n_ctx / 2`, and
    ///    accumulate over positions `[first, n_ctx - 1)`. Per chunk that is
    ///    `n_ctx - first - 1` tokens (255 at n_ctx=512), because the model's
    ///    prediction at the final position has no in-chunk target.
    ///  - `nll += -log_softmax(logits at position p)[token at p + 1]`.
    ///  - `ppl = exp(nll / count)`.
    ///
    /// The models here set `add_bos_token = false`, so llama.cpp's
    /// "replace the first token of each chunk with BOS" branch does not apply.
    ///
    /// Logits are read back in sub-batches rather than for the whole chunk at
    /// once: full-chunk logits would be 512 x 151936 floats (~311 MB), which is
    /// not a reasonable allocation on a phone. Scoring is unaffected, since only
    /// positions in the second half are ever consumed.
    /// - Parameters:
    ///   - resumeFrom: state from an earlier interrupted run. Chunks are scored
    ///     independently (the KV cache is cleared before each), so skipping the
    ///     first `chunksDone` and restoring the running `nll`/`count` reproduces
    ///     the uninterrupted result exactly.
    ///   - checkpointEvery: how often `onCheckpoint` fires, in chunks.
    func perplexity(tokens: [llama_token],
                    nCtx: Int = 512,
                    subBatch: Int = 64,
                    resumeFrom: PerplexityCheckpoint? = nil,
                    checkpointEvery: Int = 25,
                    onCheckpoint: (@Sendable (PerplexityCheckpoint) -> Void)? = nil,
                    progress: (@Sendable (Int, Int, Double) -> Void)? = nil) throws -> PerplexityResult {
        let first = nCtx / 2
        let nChunk = tokens.count / nCtx
        guard nChunk > 0 else {
            throw LlamaError.decodeFailed(-1)
        }
        let nVocab = Int(llama_vocab_n_tokens(vocab))

        var nll = 0.0
        var count = 0
        var startChunk = 0
        if let r = resumeFrom, r.chunksDone > 0, r.chunksDone < nChunk {
            nll = r.nll
            count = r.tokensScored
            startChunk = r.chunksDone
        }

        for chunk in startChunk..<nChunk {
            let start = chunk * nCtx
            clearMemory()

            // Positions [0, first) only build context; no logits needed.
            try decodeRange(tokens: tokens, from: start, to: start + first,
                            startPos: 0, wantLogits: false, nVocab: nVocab, nll: &nll, count: &count,
                            absoluteBase: start, scoreUpTo: 0)

            // Positions [first, nCtx) are scored. The final position is decoded
            // for completeness but produces no scored pair.
            var j = first
            while j < nCtx {
                let end = min(j + subBatch, nCtx)
                try decodeRange(tokens: tokens, from: start + j, to: start + end,
                                startPos: Int32(j), wantLogits: true, nVocab: nVocab,
                                nll: &nll, count: &count,
                                absoluteBase: start, scoreUpTo: nCtx - 1)
                j = end
            }

            let running = exp(nll / Double(max(count, 1)))
            progress?(chunk + 1, nChunk, running)

            let done = chunk + 1
            if done % checkpointEvery == 0 && done < nChunk {
                onCheckpoint?(PerplexityCheckpoint(chunksDone: done, totalChunks: nChunk,
                                                   nll: nll, tokensScored: count,
                                                   nCtx: nCtx, corpusTokens: tokens.count,
                                                   runningPpl: running))
            }
        }

        return PerplexityResult(ppl: exp(nll / Double(count)),
                                nll: nll,
                                tokensScored: count,
                                chunks: nChunk,
                                nCtx: nCtx,
                                resumedFromChunk: startChunk > 0 ? startChunk : nil)
    }

    /// Decodes `tokens[from..<to]` at sequential positions starting at `startPos`.
    /// When `wantLogits`, accumulates NLL for every decoded position `p` where
    /// `p < scoreUpTo`, using the ground-truth token at `p + 1`.
    private func decodeRange(tokens: [llama_token], from: Int, to: Int,
                             startPos: Int32, wantLogits: Bool, nVocab: Int,
                             nll: inout Double, count: inout Int,
                             absoluteBase: Int, scoreUpTo: Int) throws {
        guard to > from else { return }
        batchClear()
        var pos = startPos
        for idx in from..<to {
            batchAdd(tokens[idx], pos, wantLogits)
            pos += 1
        }
        let rc = llama_decode(context, batch)
        if rc != 0 { throw LlamaError.decodeFailed(rc) }

        guard wantLogits else { return }

        for k in 0..<(to - from) {
            let relPos = Int(startPos) + k          // position within the chunk
            guard relPos < scoreUpTo else { continue }
            guard let logits = llama_get_logits_ith(context, Int32(k)) else { continue }
            let target = tokens[absoluteBase + relPos + 1]
            nll += -LlamaRunner.logSoftmax(logits: logits, n: nVocab, target: Int(target))
            count += 1
        }
    }

    /// Numerically stable log-softmax at a single index, via Accelerate.
    /// Returns `log p(target)`.
    static func logSoftmax(logits: UnsafeMutablePointer<Float>, n: Int, target: Int) -> Double {
        var maxLogit: Float = 0
        vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(n))

        var shifted = [Float](repeating: 0, count: n)
        var negMax = -maxLogit
        vDSP_vsadd(logits, 1, &negMax, &shifted, 1, vDSP_Length(n))

        var cnt = Int32(n)
        var exps = [Float](repeating: 0, count: n)
        vvexpf(&exps, shifted, &cnt)

        var sumExp: Float = 0
        vDSP_sve(exps, 1, &sumExp, vDSP_Length(n))

        return Double(shifted[target]) - Double(log(sumExp))
    }

    // MARK: - Forced-choice scoring

    /// Feeds `prompt` and returns the highest-scoring token among `candidates`,
    /// judged on the logits at the first position the model would generate.
    ///
    /// This is the forced-choice form of a multiple-choice score: it reads the
    /// model's preference among exactly the legal answers rather than generating
    /// text and searching it for one. The result is therefore independent of
    /// output formatting, generation budget, and any preamble the model emits.
    /// Every question yields a prediction, so chance is 1/n and the figure is
    /// comparable across variants.
    ///
    /// Returns the argmax together with the raw logit for each candidate, so a
    /// caller can record margins or identify a model indifferent between choices.
    func argmaxOverCandidates(prompt: String,
                              candidates: [llama_token],
                              addSpecial: Bool = false,
                              parseSpecial: Bool = true) throws -> (best: llama_token, scores: [Float]) {
        precondition(!candidates.isEmpty)
        clearMemory()
        let tokens = try tokenize(prompt, addSpecial: addSpecial, parseSpecial: parseSpecial)
        _ = try evaluate(tokens: tokens, startPos: 0, logitsForLast: true)
        guard let logits = llama_get_logits_ith(context, -1) else {
            throw LlamaError.decodeFailed(-1)
        }
        let nVocab = Int(llama_vocab_n_tokens(vocab))
        var scores = [Float](repeating: -Float.infinity, count: candidates.count)
        for (i, t) in candidates.enumerated() {
            let idx = Int(t)
            if idx >= 0 && idx < nVocab { scores[i] = logits[idx] }
        }
        var bestI = 0
        for i in 1..<scores.count where scores[i] > scores[bestI] { bestI = i }
        return (candidates[bestI], scores)
    }

    /// Token id for `text`, or nil unless it encodes to exactly one token.
    /// Forced-choice scoring compares single tokens, so a multi-token answer label
    /// is out of scope for this method and returns nil for the caller to reject.
    func singleTokenId(for text: String) throws -> llama_token? {
        let t = try tokenize(text, addSpecial: false, parseSpecial: false)
        return t.count == 1 ? t[0] : nil
    }

    // MARK: - Throughput

    /// Prompt-processing and generation throughput, mirroring `llama-bench`'s
    /// defaults for this project: `n_prompt = 512`, `n_gen = 128`, `reps = 5`
    /// (verified against `tools/llama-bench/llama-bench.cpp`).
    ///
    /// The KV cache is cleared before each pass, and the two passes are timed
    /// separately so prompt-processing cost does not leak into the generation
    /// figure. Averages are taken over reps, matching llama-bench's `avg_ts`.
    func throughput(nPrompt: Int = 512, nGen: Int = 128, reps: Int = 5,
                    progress: (@Sendable (Int, Int) -> Void)? = nil) throws -> ThroughputResult {
        var ppRates: [Double] = []
        var tgRates: [Double] = []

        // llama-bench feeds token id 0 rather than real text; throughput is
        // independent of token identity and this keeps the measurement
        // independent of tokenizer behaviour.
        for rep in 0..<reps {
            // Prompt processing.
            clearMemory()
            var t0 = DispatchTime.now().uptimeNanoseconds
            var remaining = nPrompt
            var pos: Int32 = 0
            while remaining > 0 {
                let n = min(remaining, Int(batchCapacity))
                batchClear()
                for _ in 0..<n {
                    batchAdd(0, pos, false)
                    pos += 1
                }
                let rc = llama_decode(context, batch)
                if rc != 0 { throw LlamaError.decodeFailed(rc) }
                remaining -= n
            }
            llama_synchronize(context)
            var t1 = DispatchTime.now().uptimeNanoseconds
            ppRates.append(Double(nPrompt) / (Double(t1 - t0) / 1e9))

            // Generation: one token per decode, as in llama-bench's tg pass.
            clearMemory()
            t0 = DispatchTime.now().uptimeNanoseconds
            for i in 0..<nGen {
                batchClear()
                batchAdd(0, Int32(i), true)
                let rc = llama_decode(context, batch)
                if rc != 0 { throw LlamaError.decodeFailed(rc) }
            }
            llama_synchronize(context)
            t1 = DispatchTime.now().uptimeNanoseconds
            tgRates.append(Double(nGen) / (Double(t1 - t0) / 1e9))

            progress?(rep + 1, reps)
        }

        clearMemory()
        return ThroughputResult(
            promptTokensPerSec: ppRates.reduce(0, +) / Double(ppRates.count),
            genTokensPerSec: tgRates.reduce(0, +) / Double(tgRates.count),
            promptSamples: ppRates,
            genSamples: tgRates,
            nPrompt: nPrompt, nGen: nGen, reps: reps)
    }
}

struct PerplexityResult {
    var ppl: Double
    var nll: Double
    var tokensScored: Int
    var chunks: Int
    var nCtx: Int
    var resumedFromChunk: Int?
}

/// Resumable state for a perplexity run.
///
/// A full 583-chunk pass takes tens of minutes per variant, so an interruption
/// part-way through is expensive to redo. Because chunks are scored
/// independently, the running `nll` and `tokensScored` are sufficient state:
/// resuming skips completed chunks and continues the same accumulation.
///
/// `corpusTokens` and `nCtx` are recorded so a checkpoint is only reused for the
/// same corpus and chunking; `BenchmarkSuite` additionally requires a matching
/// model SHA-256 before resuming.
struct PerplexityCheckpoint: Codable {
    var chunksDone: Int
    var totalChunks: Int
    var nll: Double
    var tokensScored: Int
    var nCtx: Int
    var corpusTokens: Int
    var runningPpl: Double
    var modelSHA256: String?
    var tag: String?
    var savedAt: String?
}

struct ThroughputResult {
    var promptTokensPerSec: Double
    var genTokensPerSec: Double
    var promptSamples: [Double]
    var genSamples: [Double]
    var nPrompt: Int
    var nGen: Int
    var reps: Int
}

import Foundation
import llama

/// Thin Swift wrapper over the llama.cpp C API.
///
/// Written against llama.cpp b10201 (`llama_model_load_from_file` /
/// `llama_init_from_model` era), with C symbol names taken from `include/llama.h`
/// at that revision. These names have changed across upstream releases, so after
/// a llama.cpp bump the header is the reference for updating them.
enum LlamaError: Error, CustomStringConvertible {
    case modelLoadFailed(String)
    case contextInitFailed
    case tokenizeFailed(Int32)
    case decodeFailed(Int32)

    var description: String {
        switch self {
        case .modelLoadFailed(let p): return "llama_model_load_from_file failed for \(p)"
        case .contextInitFailed:      return "llama_init_from_model returned null"
        case .tokenizeFailed(let n):  return "llama_tokenize failed (returned \(n))"
        case .decodeFailed(let n):    return "llama_decode failed (returned \(n))"
        }
    }
}

/// Captures llama.cpp's log output for display in-app.
///
/// llama.cpp's log is the record of whether a tensor received a KleidiAI kernel
/// or used the stock CPU path ("kleidiai: no kernel for tensor type ..."). A
/// device provides no stderr to read, so the log is captured here instead.
final class LlamaLog: @unchecked Sendable {
    static let shared = LlamaLog()
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ s: String) {
        lock.lock(); defer { lock.unlock() }
        // llama.cpp emits partial lines; join then split so entries stay whole.
        lines.append(s)
        if lines.count > 4000 { lines.removeFirst(lines.count - 4000) }
    }

    func drainAll() -> String {
        lock.lock(); defer { lock.unlock() }
        return lines.joined()
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        lines.removeAll()
    }

    /// Lines mentioning KleidiAI, which is what the Arm-acceleration claim rests on.
    func kleidiaiLines() -> [String] {
        drainAll()
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
            .filter { $0.lowercased().contains("kleidiai") }
    }
}

/// Installs the log hook. Must be a capture-free C function pointer.
func installLlamaLogHook() {
    llama_log_set({ (_ level: ggml_log_level, text: UnsafePointer<CChar>?, _ userData: UnsafeMutableRawPointer?) in
        guard let text else { return }
        LlamaLog.shared.append(String(cString: text))
    }, nil)
}

/// How much of the model to place on the GPU.
///
/// Stated explicitly rather than defaulted, because it determines which backend
/// the benchmark measures. llama.cpp offloads to Metal by default on iOS, and
/// with Metal handling the matmuls the CPU backend -- and therefore KleidiAI --
/// does not run. Figures reported as Arm-CPU or KleidiAI results come from
/// `.cpuOnly`.
enum Backend {
    case cpuOnly        // n_gpu_layers = 0 -- the Arm CPU + KleidiAI path
    case metal(Int32)   // offload N layers to Metal

    var nGpuLayers: Int32 {
        switch self {
        case .cpuOnly:       return 0
        case .metal(let n):  return n
        }
    }

    var label: String {
        switch self {
        case .cpuOnly:      return "CPU (Arm/KleidiAI eligible)"
        case .metal(let n): return "Metal (n_gpu_layers=\(n))"
        }
    }
}

actor LlamaRunner {
    internal var model: OpaquePointer
    internal var context: OpaquePointer
    internal var vocab: OpaquePointer
    internal var batch: llama_batch
    internal let batchCapacity: Int32

    let modelPath: String
    let backend: Backend

    private init(model: OpaquePointer,
                 context: OpaquePointer,
                 vocab: OpaquePointer,
                 batchCapacity: Int32,
                 modelPath: String,
                 backend: Backend) {
        self.model = model
        self.context = context
        self.vocab = vocab
        self.batchCapacity = batchCapacity
        self.batch = llama_batch_init(batchCapacity, 0, 1)
        self.modelPath = modelPath
        self.backend = backend
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
    }

    /// `llama_backend_init()` is process-global; call once per process.
    private static var backendReady = false
    private static func ensureBackend() {
        if !backendReady {
            installLlamaLogHook()
            llama_backend_init()
            backendReady = true
        }
    }

    static func load(path: String,
                     backend: Backend = .cpuOnly,
                     nCtx: UInt32 = 2048,
                     nBatch: UInt32 = 512,
                     nThreads: Int32? = nil) throws -> LlamaRunner {
        ensureBackend()

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = backend.nGpuLayers

        guard let model = llama_model_load_from_file(path, mparams) else {
            throw LlamaError.modelLoadFailed(path)
        }

        // Apple silicon mixes P and E cores; llama.cpp's own iOS example leaves
        // two aside. The same heuristic is used here to keep throughput from
        // being dominated by scheduling noise.
        let threads = nThreads ?? Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))

        var cparams = llama_context_default_params()
        cparams.n_ctx           = nCtx
        cparams.n_batch         = nBatch
        cparams.n_threads       = threads
        cparams.n_threads_batch = threads

        guard let context = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            throw LlamaError.contextInitFailed
        }

        guard let vocab = llama_model_get_vocab(model) else {
            llama_free(context)
            llama_model_free(model)
            throw LlamaError.contextInitFailed
        }

        return LlamaRunner(model: model,
                           context: context,
                           vocab: vocab,
                           batchCapacity: Int32(nBatch),
                           modelPath: path,
                           backend: backend)
    }

    // MARK: - Model introspection

    func describe() -> String {
        var buf = [CChar](repeating: 0, count: 256)
        let n = llama_model_desc(model, &buf, 256)
        let desc = n > 0 ? String(cString: buf) : "(unknown)"
        let sizeBytes = llama_model_size(model)
        let params = llama_model_n_params(model)
        return """
        desc:    \(desc)
        params:  \(String(format: "%.3f B", Double(params) / 1e9))
        weights: \(String(format: "%.1f MB", Double(sizeBytes) / 1e6))
        n_ctx:   \(llama_n_ctx(context))
        vocab:   \(llama_vocab_n_tokens(vocab))
        backend: \(backend.label)
        """
    }

    var contextSize: UInt32 { llama_n_ctx(context) }
    var vocabSize: Int32 { llama_vocab_n_tokens(vocab) }
    var paramCount: UInt64 { llama_model_n_params(model) }
    var weightsBytes: UInt64 { llama_model_size(model) }

    // MARK: - Tokenization

    /// - Parameter parseSpecial: when true, control tokens written literally in
    ///   `text` (`<|im_start|>`, `<think>`, ...) are matched as single special
    ///   tokens. It must be true for any chat-template-formatted prompt: with it
    ///   false, `<|im_start|>` tokenizes as the six pieces `<`, `|`, `im`,
    ///   `_start`, `|`, `>` and the model receives literal text where markup was
    ///   intended. It stays false for plain corpus text, matching
    ///   `common_tokenize`'s default, which is what llama-perplexity uses.
    func tokenize(_ text: String,
                  addSpecial: Bool = true,
                  parseSpecial: Bool = false) throws -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        // Upper bound: one token per byte, plus room for a BOS.
        var capacity = utf8Count + (addSpecial ? 1 : 0) + 1
        var tokens = [llama_token](repeating: 0, count: Int(capacity))

        var n = llama_tokenize(vocab, text, utf8Count, &tokens, capacity, addSpecial, parseSpecial)
        if n < 0 {
            // Negative return is the required capacity.
            capacity = -n
            tokens = [llama_token](repeating: 0, count: Int(capacity))
            n = llama_tokenize(vocab, text, utf8Count, &tokens, capacity, addSpecial, parseSpecial)
            if n < 0 { throw LlamaError.tokenizeFailed(n) }
        }
        return Array(tokens[0..<Int(n)])
    }

    func piece(for token: llama_token) -> String {
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
            if n < 0 { return "" }
        }
        let bytes = buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Low-level decode helpers

    func clearMemory() {
        llama_memory_clear(llama_get_memory(context), true)
    }

    internal func batchClear() { batch.n_tokens = 0 }

    internal func batchAdd(_ token: llama_token, _ pos: llama_pos, _ wantLogits: Bool) {
        let i = Int(batch.n_tokens)
        batch.token[i]    = token
        batch.pos[i]      = pos
        batch.n_seq_id[i] = 1
        batch.seq_id[i]![0] = 0
        batch.logits[i]   = wantLogits ? 1 : 0
        batch.n_tokens += 1
    }

    /// Feeds `tokens` starting at `startPos`, in chunks that fit the batch.
    /// Only the final token of the final chunk requests logits.
    @discardableResult
    func evaluate(tokens: [llama_token], startPos: Int32, logitsForLast: Bool) throws -> Int32 {
        var pos = startPos
        var i = 0
        while i < tokens.count {
            let chunk = min(Int(batchCapacity), tokens.count - i)
            batchClear()
            for j in 0..<chunk {
                let isLast = (i + j == tokens.count - 1)
                batchAdd(tokens[i + j], pos, logitsForLast && isLast)
                pos += 1
            }
            let rc = llama_decode(context, batch)
            if rc != 0 { throw LlamaError.decodeFailed(rc) }
            i += chunk
        }
        llama_synchronize(context)
        return pos
    }

    func logits(at i: Int32) -> UnsafeMutablePointer<Float>? {
        llama_get_logits_ith(context, i)
    }

    func isEndOfGeneration(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocab, token)
    }

    // MARK: - Greedy generation

    /// Generates up to `maxTokens` greedily (temperature 0) from `prompt`.
    /// Greedy decoding keeps reported metrics reproducible across runs and
    /// matches the sampling used by the desktop scripts.
    /// Why generation stopped. Distinguishes a model that finished its answer from
    /// one cut off at the cap, which a caller cannot infer from the text alone.
    enum StopReason: String, Sendable {
        case endOfSequence
        case tokenLimit
    }

    struct Generation: Sendable {
        var text: String
        var tokens: Int
        var stopReason: StopReason
    }

    func generateDetailed(prompt: String, maxTokens: Int,
                          addSpecial: Bool = true,
                          parseSpecial: Bool = false) throws -> Generation {
        clearMemory()

        let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        defer { llama_sampler_free(sampler) }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())

        let promptTokens = try tokenize(prompt, addSpecial: addSpecial, parseSpecial: parseSpecial)
        var pos = try evaluate(tokens: promptTokens, startPos: 0, logitsForLast: true)

        var out = ""
        var produced = 0
        var stop = StopReason.tokenLimit
        // -1 samples from the logits of the last token in the most recent batch.
        var next = llama_sampler_sample(sampler, context, -1)

        while produced < maxTokens {
            if isEndOfGeneration(next) {
                stop = .endOfSequence
                break
            }
            out += piece(for: next)
            produced += 1

            batchClear()
            batchAdd(next, pos, true)
            pos += 1
            let rc = llama_decode(context, batch)
            if rc != 0 { throw LlamaError.decodeFailed(rc) }
            next = llama_sampler_sample(sampler, context, -1)
        }
        llama_synchronize(context)
        return Generation(text: out, tokens: produced, stopReason: stop)
    }

    /// Tuple-returning form kept for callers that do not need the stop reason.
    func generate(prompt: String, maxTokens: Int,
                  addSpecial: Bool = true,
                  parseSpecial: Bool = false) throws -> (text: String, tokensGenerated: Int) {
        let g = try generateDetailed(prompt: prompt, maxTokens: maxTokens,
                                     addSpecial: addSpecial, parseSpecial: parseSpecial)
        return (g.text, g.tokens)
    }
}

import Foundation

/// One result record per model variant.
///
/// The first eleven keys match `scripts/benchmark.py`'s `record` dict exactly,
/// since `scripts/summarize_results.py` reads them by name with no remapping.
/// The remaining keys are additions: `summarize_results.py` ignores unknown
/// keys, and recording which file and backend produced a measurement makes a
/// result self-describing rather than dependent on surrounding context.
struct BenchmarkRecord: Encodable {
    // --- schema shared with scripts/benchmark.py ---
    var tag: String
    var device: String
    var model_path: String
    var timestamp_utc: String
    var disk_bytes: Int64
    var peak_ram_bytes: UInt64
    var prompt_tokens_per_sec: Double?
    var gen_tokens_per_sec: Double?
    var perplexity: Double?
    /// Forced-choice accuracy over A/B/C/D is the reported instruction metric; see
    /// InstructionEval.Outcome. Parse rate and free-generation accuracy sit under
    /// `harness` as diagnostics.
    var instruction_forced_choice_accuracy: Double?
    var instruction_per_subject_forced_choice_accuracy: [String: Double]?
    var llama_bench_raw: String?          // always null on-device; desktop-only field

    // --- additions ---
    var model_sha256: String?
    var backend: String
    var harness: Harness

    struct Harness: Encodable {
        var app: String
        var llama_cpp_build: String
        var arm_arch: String
        var threads: Int
        var n_ctx: UInt32

        var perplexity_method: String
        var perplexity_corpus: String
        var perplexity_chunks: Int?
        var perplexity_tokens_scored: Int?

        var throughput_method: String
        var throughput_prompt_samples: [Double]?
        var throughput_gen_samples: [Double]?

        var instruction_eval_prompt_format: String
        var instruction_eval_total: Int?
        var instruction_eval_method: String?
        var instruction_eval_forced_choice_correct: Int?
        var instruction_eval_forced_choice_predictions: String?
        var instruction_eval_forced_choice_histogram: [String: Int]?
        // Diagnostics from the generate-and-parse path.
        var instruction_eval_parse_rate_diagnostic: Double?
        var instruction_eval_parsed: Int?
        var instruction_eval_accuracy_diagnostic: Double?
        var instruction_eval_per_subject_accuracy_diagnostic: [String: Double]?
        var instruction_eval_metric_note: String?

        var peak_ram_scope: String
    }

    static func timestamp() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: Date())
    }

    func encodedJSON() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    enum CodingKeys: String, CodingKey {
        case tag, device, model_path, timestamp_utc, disk_bytes, peak_ram_bytes
        case prompt_tokens_per_sec, gen_tokens_per_sec, perplexity
        case instruction_forced_choice_accuracy
        case instruction_per_subject_forced_choice_accuracy
        case llama_bench_raw
        case model_sha256, backend, harness
    }

    /// Written by hand so the twelve schema keys are always present, emitting an
    /// explicit `null` where a metric was skipped. Swift's synthesised encoder
    /// omits nil optionals entirely, which would make the on-device record a
    /// subset of `scripts/benchmark.py`'s rather than the same shape. Anything
    /// consuming both should not have to distinguish "absent" from "null".
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(tag, forKey: .tag)
        try c.encode(device, forKey: .device)
        try c.encode(model_path, forKey: .model_path)
        try c.encode(timestamp_utc, forKey: .timestamp_utc)
        try c.encode(disk_bytes, forKey: .disk_bytes)
        try c.encode(peak_ram_bytes, forKey: .peak_ram_bytes)

        // Optional metrics: encode the value or an explicit null.
        try encodeOrNull(&c, prompt_tokens_per_sec, .prompt_tokens_per_sec)
        try encodeOrNull(&c, gen_tokens_per_sec, .gen_tokens_per_sec)
        try encodeOrNull(&c, perplexity, .perplexity)
        try encodeOrNull(&c, instruction_forced_choice_accuracy, .instruction_forced_choice_accuracy)
        try encodeOrNull(&c, instruction_per_subject_forced_choice_accuracy,
                         .instruction_per_subject_forced_choice_accuracy)
        // Desktop-only field: never populated on-device, always an explicit null.
        try c.encodeNil(forKey: .llama_bench_raw)

        try encodeOrNull(&c, model_sha256, .model_sha256)
        try c.encode(backend, forKey: .backend)
        try c.encode(harness, forKey: .harness)
    }

    private func encodeOrNull<T: Encodable>(_ c: inout KeyedEncodingContainer<CodingKeys>,
                                           _ value: T?, _ key: CodingKeys) throws {
        if let value { try c.encode(value, forKey: key) } else { try c.encodeNil(forKey: key) }
    }
}

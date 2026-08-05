import Foundation

/// Runs one prompt through each model variant in turn and reports each response.
///
/// Variants are loaded **sequentially**, with the previous model released before
/// the next is loaded. fp16 alone peaks near 4.2 GB resident on device, so holding
/// two or three models at once risks termination under memory pressure. The
/// Sequential ordering also gives the per-panel progress a precise meaning:
/// exactly one model is resident at any moment.
///
/// This is an exploration tool. It does not write to `results/` and is not part of
/// the benchmark path, so it carries none of the record schema. Model loading and
/// generation come from `LlamaRunner`; nothing here re-implements them.
enum PromptComparison {

    enum Phase: Sendable {
        case loading
        case generating
        case finished
        case failed(String)
    }

    struct Response: Sendable {
        var variant: ModelVariant
        var text: String
        var tokens: Int
        var loadSeconds: Double
        var generateSeconds: Double
        /// Whole-process resident size sampled just before this variant was loaded.
        /// Across a run it shows the previous model being released: it returns near
        /// its baseline between variants rather than accumulating.
        var residentBytesBeforeLoad: UInt64
        var stopReason: LlamaRunner.StopReason
        /// Read from the loaded model, so a panel reports what it actually ran
        /// rather than a value copied into the UI.
        var paramCount: UInt64
        var weightsBytes: UInt64
        /// Generation rate for this response. Cheap to derive from the same timing
        /// and useful next to the text, though it is a single short sample rather
        /// than the 5-rep measurement the benchmark path reports.
        var tokensPerSec: Double
    }

    /// - Parameters:
    ///   - maxTokens: response cap. A cap is required because not every variant
    ///     emits an end-of-turn token; see CompareState.maxTokens.
    ///   - onPhase: fires as each variant enters a phase, so a caller can show
    ///     per-variant progress.
    ///   - onResponse: fires once per variant that produced a response.
    static func run(prompt: String,
                    variants: [ModelVariant] = ModelVariant.allCases,
                    maxTokens: Int = 256,
                    backend: Backend = .cpuOnly,
                    onPhase: @escaping @Sendable (ModelVariant, Phase) -> Void,
                    onResponse: @escaping @Sendable (Response) -> Void) async {

        let templated = ChatTemplate.qwen3NoThink(prompt)

        for variant in variants {
            onPhase(variant, .loading)

            guard let url = ResourceLocator.locate(variant) else {
                onPhase(variant, .failed("\(variant.fileName) not found on this device"))
                continue
            }

            do {
                let residentBefore = MemorySampler.residentBytes()
                let loadStart = Date()
                // Loading is synchronous and takes seconds for fp16, so it runs off
                // the main actor to keep the UI responsive while it happens.
                let path = url.path
                let runner = try await Task.detached(priority: .userInitiated) {
                    try LlamaRunner.load(path: path, backend: backend)
                }.value
                let loadSeconds = Date().timeIntervalSince(loadStart)

                onPhase(variant, .generating)

                let params = await runner.paramCount
                let weights = await runner.weightsBytes

                let genStart = Date()
                // parseSpecial: true — the templated prompt carries control tokens.
                let g = try await runner.generateDetailed(prompt: templated,
                                                         maxTokens: maxTokens,
                                                         addSpecial: false,
                                                         parseSpecial: true)
                let (text, tokens) = (g.text, g.tokens)
                let generateSeconds = Date().timeIntervalSince(genStart)

                onResponse(Response(
                    variant: variant,
                    text: text,
                    tokens: tokens,
                    loadSeconds: loadSeconds,
                    generateSeconds: generateSeconds,
                    residentBytesBeforeLoad: residentBefore,
                    stopReason: g.stopReason,
                    paramCount: params,
                    weightsBytes: weights,
                    tokensPerSec: generateSeconds > 0 ? Double(tokens) / generateSeconds : 0))
                onPhase(variant, .finished)

                // `runner` holds the only reference and goes out of scope at the end
                // of this block, freeing the model and context before the next
                // variant is loaded.
            } catch {
                onPhase(variant, .failed(error.localizedDescription))
            }
        }
    }
}

import Foundation

/// Minimal end-to-end check: load a GGUF, tokenize, decode, generate a few
/// tokens, tear down. Passing this is the precondition for the full benchmark
/// suite, which collects its metrics through the same load and decode path.
struct SmokeTest {
    struct Result {
        var variant: ModelVariant
        var modelPath: String
        var diskBytes: Int64
        var modelInfo: String
        var promptTokenCount: Int
        var generated: String
        var tokensGenerated: Int
        var loadSeconds: Double
        var generateSeconds: Double
        var kleidiaiLogLines: [String]
    }

    static let prompt = "The capital of France is"

    static func run(variant: ModelVariant,
                    backend: Backend,
                    maxTokens: Int = 16,
                    log: @escaping (String) -> Void) async -> Swift.Result<Result, Error> {
        guard let url = ResourceLocator.locate(variant) else {
            let msg = """
            Could not find \(variant.fileName).
            Searched:
              \(ResourceLocator.searchPaths.joined(separator: "\n  "))
            """
            log(msg)
            return .failure(NSError(domain: "LlamaBench", code: 2,
                                    userInfo: [NSLocalizedDescriptionKey: msg]))
        }

        let diskBytes = ResourceLocator.fileSize(url)
        log("model:   \(url.lastPathComponent)")
        log("path:    \(url.path)")
        log("disk:    \(String(format: "%.1f MB", Double(diskBytes) / 1e6))")
        log("backend: \(backend.label)")

        LlamaLog.shared.clear()

        do {
            let t0 = Date()
            let runner = try LlamaRunner.load(path: url.path, backend: backend)
            let loadSeconds = Date().timeIntervalSince(t0)
            log("loaded in \(String(format: "%.2f s", loadSeconds))")

            let info = await runner.describe()
            log(info)

            let promptTokens = try await runner.tokenize(SmokeTest.prompt)
            log("prompt \"\(SmokeTest.prompt)\" -> \(promptTokens.count) tokens")

            let t1 = Date()
            let (text, produced) = try await runner.generate(prompt: SmokeTest.prompt,
                                                            maxTokens: maxTokens)
            let genSeconds = Date().timeIntervalSince(t1)

            log("generated \(produced) tokens in \(String(format: "%.2f s", genSeconds)) "
                + "(\(String(format: "%.1f tok/s", Double(produced) / max(genSeconds, 1e-9))))")
            log("output:  \(text.isEmpty ? "(empty)" : text)")

            let kai = LlamaLog.shared.kleidiaiLines()
            if kai.isEmpty {
                log("kleidiai: no KleidiAI log lines emitted for this model.")
            } else {
                log("kleidiai: \(kai.count) log line(s); first few:")
                for l in kai.prefix(4) { log("  \(l)") }
            }

            return .success(Result(variant: variant,
                                   modelPath: url.path,
                                   diskBytes: diskBytes,
                                   modelInfo: info,
                                   promptTokenCount: promptTokens.count,
                                   generated: text,
                                   tokensGenerated: produced,
                                   loadSeconds: loadSeconds,
                                   generateSeconds: genSeconds,
                                   kleidiaiLogLines: kai))
        } catch {
            log("FAILED: \(error)")
            return .failure(error)
        }
    }
}

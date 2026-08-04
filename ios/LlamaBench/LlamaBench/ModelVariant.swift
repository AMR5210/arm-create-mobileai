import Foundation

/// The three models the benchmark compares.
///
/// `tag` values are load-bearing: `scripts/summarize_results.py` reads
/// `results/<tag>.json` by exactly these names, so on-device output feeds the
/// existing desktop pipeline without Python-side changes.
enum ModelVariant: String, CaseIterable, Identifiable {
    case baselineFP16 = "baseline-fp16"
    case ptq2bit      = "ptq-2bit"
    case qat2bit      = "qat-2bit"

    var id: String { rawValue }
    var tag: String { rawValue }

    var fileName: String {
        switch self {
        case .baselineFP16: return "qwen3-0.6b-fp16.gguf"
        case .ptq2bit:      return "qwen3-0.6b-ptq-q2_k.gguf"
        case .qat2bit:      return "qwen3-0.6b-qat-q2_k.gguf"
        }
    }

    var displayName: String {
        switch self {
        case .baselineFP16: return "Baseline fp16"
        case .ptq2bit:      return "PTQ 2-bit"
        case .qat2bit:      return "QAT 2-bit"
        }
    }
}

/// Finds model and eval files across the three locations they can occupy,
/// in priority order:
///   1. App bundle -- how models ship for an on-device benchmark run.
///   2. Documents  -- files pushed via Files.app / iTunes sharing, which avoids
///      a 2.4 GB app bundle during iteration.
///   3. `LLAMABENCH_MODEL_DIR` -- a host directory, readable from the simulator,
///      letting the desktop-side smoke test use the repo's models/ directory
///      directly.
///
/// The environment variable keeps local absolute paths out of this public
/// repository.
struct ResourceLocator {
    static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static var envSearchDirectory: URL? {
        guard let p = ProcessInfo.processInfo.environment["LLAMABENCH_MODEL_DIR"],
              !p.isEmpty else { return nil }
        return URL(fileURLWithPath: p, isDirectory: true)
    }

    /// All directories searched, in priority order, for diagnostics.
    static var searchPaths: [String] {
        var out: [String] = []
        if let b = Bundle.main.resourceURL { out.append("bundle: \(b.path)") }
        out.append("documents: \(documentsDirectory.path)")
        if let e = envSearchDirectory { out.append("env LLAMABENCH_MODEL_DIR: \(e.path)") }
        else { out.append("env LLAMABENCH_MODEL_DIR: (unset)") }
        return out
    }

    static func locate(_ fileName: String) -> URL? {
        let base = (fileName as NSString).deletingPathExtension
        let ext  = (fileName as NSString).pathExtension

        if let u = Bundle.main.url(forResource: base, withExtension: ext.isEmpty ? nil : ext),
           FileManager.default.fileExists(atPath: u.path) {
            return u
        }
        let docs = documentsDirectory.appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: docs.path) { return docs }

        if let env = envSearchDirectory {
            let candidate = env.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    static func locate(_ variant: ModelVariant) -> URL? { locate(variant.fileName) }

    static func fileSize(_ url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int64 else { return 0 }
        return size
    }
}

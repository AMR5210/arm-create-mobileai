import SwiftUI

@MainActor
final class BenchState: ObservableObject {
    @Published var logText: String = ""
    @Published var isRunning: Bool = false
    @Published var selectedVariant: ModelVariant = .ptq2bit
    @Published var useCPUOnly: Bool = true

    func log(_ s: String) {
        logText += s + "\n"
        // Mirrored to stdout so a headless run driven by
        // `simctl launch --console-pty` (or an attached device via idevicesyslog)
        // produces the same transcript as the on-screen log, making benchmark
        // runs scriptable as well as interactive.
        print(s)
    }

    func clear() { logText = "" }

    /// Runs the full five-metric suite for one variant and writes its JSON record.
    func runBenchmark(variant: ModelVariant) async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        let env = ProcessInfo.processInfo.environment
        var opts = BenchmarkSuite.Options()
        opts.backend = useCPUOnly ? .cpuOnly : .metal(99)
        opts.device = env["LLAMABENCH_DEVICE"] ?? Self.defaultDeviceLabel
        if let n = env["LLAMABENCH_PPL_CHUNKS"], let v = Int(n) { opts.perplexityChunkLimit = v }
        if env["LLAMABENCH_SKIP_PPL"] == "1" { opts.skipPerplexity = true }
        if env["LLAMABENCH_SKIP_INSTR"] == "1" { opts.skipInstructionEval = true }
        if env["LLAMABENCH_SKIP_TP"] == "1" { opts.skipThroughput = true }
        if let r = env["LLAMABENCH_REPS"], let v = Int(r) { opts.throughputReps = v }
        if let n = env["LLAMABENCH_INSTR_LIMIT"], let v = Int(n) { opts.instructionEvalLimit = v }
        if let n = env["LLAMABENCH_INSTR_DEBUG"], let v = Int(n) { opts.instructionEvalDebugFirstN = v }
        if let n = env["LLAMABENCH_INSTR_TOKENS"], let v = Int(n) { opts.instructionEvalMaxTokens = v }
        if let n = env["LLAMABENCH_INSTR_STRIDE"], let v = Int(n) { opts.instructionEvalStride = v }
        if let n = env["LLAMABENCH_CKPT_EVERY"], let v = Int(n) { opts.checkpointEvery = v }
        if env["LLAMABENCH_NO_RESUME"] == "1" { opts.resumeFromCheckpoint = false }
        if let name = env["LLAMABENCH_OUT_NAME"], !name.isEmpty { opts.outputBasename = name }
        opts.outputDirectory = ResourceLocator.outputDirectory

        do {
            let record = try await BenchmarkSuite.run(variant: variant, options: opts) { line in
                Task { @MainActor in self.log(line) }
            }
            let out = try BenchmarkSuite.write(record, to: opts.outputDirectory,
                                              basename: opts.outputBasename)
            log("wrote \(out.path)")
            if let json = String(data: try record.encodedJSON(), encoding: .utf8) {
                log("--- \(record.tag).json ---")
                log(json)
            }
            log("=== \(variant.tag) COMPLETE ===\n")
        } catch {
            log("=== \(variant.tag) FAILED: \(error.localizedDescription) ===\n")
        }
    }

    /// Simulator runs must not be mistaken for device runs; the device string is
    /// what distinguishes them in the results table.
    static var defaultDeviceLabel: String {
        #if targetEnvironment(simulator)
        return "iOS Simulator (desktop host) - DEV ONLY, not a reportable device"
        #else
        return UIDevice.current.model
        #endif
    }

    /// Runs without UI interaction when LLAMABENCH_AUTORUN=1.
    ///
    /// Overrides: LLAMABENCH_MODE (smoke|bench), LLAMABENCH_VARIANT
    /// (baseline-fp16|ptq-2bit|qat-2bit|all), LLAMABENCH_BACKEND (cpu|metal),
    /// LLAMABENCH_PPL_CHUNKS, LLAMABENCH_SKIP_PPL, LLAMABENCH_SKIP_INSTR,
    /// LLAMABENCH_REPS, LLAMABENCH_OUT_DIR, LLAMABENCH_DEVICE.
    func autorunIfRequested() async {
        let env = ProcessInfo.processInfo.environment
        guard env["LLAMABENCH_AUTORUN"] == "1" else { return }

        if let b = env["LLAMABENCH_BACKEND"] {
            useCPUOnly = (b.lowercased() != "metal")
        }
        let variantArg = env["LLAMABENCH_VARIANT"] ?? "ptq-2bit"
        let variants: [ModelVariant] = variantArg == "all"
            ? ModelVariant.allCases
            : (ModelVariant(rawValue: variantArg).map { [$0] } ?? [.ptq2bit])

        log("### autorun (LLAMABENCH_AUTORUN=1)")
        showEnvironment()

        if env["LLAMABENCH_MODE"] == "bench" {
            for v in variants {
                selectedVariant = v
                await runBenchmark(variant: v)
            }
        } else {
            for v in variants {
                selectedVariant = v
                await runSmokeTest()
            }
        }
        log("### autorun complete")
    }

    func showEnvironment() {
        log("=== resource search paths ===")
        for p in ResourceLocator.searchPaths { log("  \(p)") }
        log("=== model availability ===")
        for v in ModelVariant.allCases {
            if let u = ResourceLocator.locate(v) {
                let mb = Double(ResourceLocator.fileSize(u)) / 1e6
                log("  [found]   \(v.tag): \(u.lastPathComponent) (\(String(format: "%.1f MB", mb)))")
            } else {
                log("  [MISSING] \(v.tag): \(v.fileName)")
            }
        }
        log("")
    }

    func runSmokeTest() async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        let backend: Backend = useCPUOnly ? .cpuOnly : .metal(99)
        log("=== smoke test: \(selectedVariant.displayName) ===")
        let result = await SmokeTest.run(variant: selectedVariant,
                                        backend: backend) { [weak self] line in
            Task { @MainActor in self?.log(line) }
        }
        switch result {
        case .success:
            log("=== PASS ===\n")
        case .failure(let e):
            log("=== FAIL: \(e.localizedDescription) ===\n")
        }
    }
}

struct ContentView: View {
    @StateObject private var state = BenchState()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("LlamaBench")
                .font(.title2).bold()
            Text("On-device 2-bit QAT vs PTQ benchmark harness")
                .font(.caption).foregroundStyle(.secondary)

            Picker("Model", selection: $state.selectedVariant) {
                ForEach(ModelVariant.allCases) { v in
                    Text(v.displayName).tag(v)
                }
            }
            .pickerStyle(.segmented)
            .disabled(state.isRunning)

            Toggle("CPU only (Arm / KleidiAI path)", isOn: $state.useCPUOnly)
                .disabled(state.isRunning)
                .font(.callout)

            HStack {
                Button("Environment") { state.showEnvironment() }
                    .disabled(state.isRunning)
                Button("Smoke test") {
                    Task { await state.runSmokeTest() }
                }
                .disabled(state.isRunning)
                Button(state.isRunning ? "Running…" : "Benchmark") {
                    Task { await state.runBenchmark(variant: state.selectedVariant) }
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.isRunning)
                Button("Clear") { state.clear() }
                    .disabled(state.isRunning)
            }

            ScrollView {
                Text(state.logText.isEmpty ? "No output yet." : state.logText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .background(Color.secondary.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
        .task { await state.autorunIfRequested() }
    }
}

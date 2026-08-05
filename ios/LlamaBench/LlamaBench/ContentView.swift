import SwiftUI
import UIKit

@MainActor
final class BenchState: ObservableObject {
    @Published var logText: String = ""
    @Published var isRunning: Bool = false
    /// Responses from a headless compare run, kept for match detection at the end.
    private var compareResponses: [String: String] = [:]
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

    /// Holds the screen awake for the duration of a run.
    ///
    /// A full pass takes tens of minutes per variant. If the device locks, the app
    /// stops being frontmost and its work is suspended, which stalls the run and
    /// distorts the peak-RAM and throughput figures it was measuring. This is a
    /// safety net independent of the Auto-Lock setting.
    ///
    /// Scoped to an active run rather than left on: iOS also clears the flag when
    /// the app stops being frontmost, and a `willTerminate` observer resets it if
    /// the app is killed mid-run.
    private func setIdleTimerDisabled(_ disabled: Bool) {
        // ScreenWake returns the state UIKit holds afterwards, so the log records
        // the effective value rather than the requested one.
        log("idle timer disabled: \(ScreenWake.setDisabled(disabled))")
    }

    init() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification,
            object: nil, queue: .main
        ) { _ in
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    /// Runs the full five-metric suite for one variant and writes its JSON record.
    func runBenchmark(variant: ModelVariant) async {
        guard !isRunning else { return }
        isRunning = true
        setIdleTimerDisabled(true)
        defer {
            isRunning = false
            setIdleTimerDisabled(false)
        }

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

    /// See DeviceInfo: the record needs the specific model rather than the device
    /// family, and a simulator label states that it is not a reportable device.
    static var defaultDeviceLabel: String { DeviceInfo.recordLabel }

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

        if env["LLAMABENCH_MODE"] == "compare" {
            let prompt = env["LLAMABENCH_PROMPT"] ?? "What is the capital of France? Answer in one sentence."
            log("### compare: \(prompt.debugDescription)")
            await PromptComparison.run(
                prompt: prompt,
                maxTokens: Int(env["LLAMABENCH_COMPARE_TOKENS"] ?? "") ?? 256,
                onPhase: { [weak self] v, phase in
                    Task { @MainActor in
                        switch phase {
                        case .loading:             self?.log("  \(v.tag): loading")
                        case .generating:          self?.log("  \(v.tag): generating")
                        case .finished:            break
                        case .failed(let message): self?.log("  \(v.tag): FAILED \(message)")
                        }
                    }
                },
                onResponse: { [weak self] r in
                    Task { @MainActor in
                        let f = ResponseAnalysis.flags(text: r.text, stopReason: r.stopReason)
                        var tags: [String] = []
                        if f.repetition {
                            tags.append("REPETITION(unit=\(f.repeatUnitWords) x\(f.repeatCount))")
                        }
                        if f.truncated { tags.append("TRUNCATED") }
                        self?.log(String(format: "  %@: %.3fB · %.0f MB · %d tok, %.1f tok/s, "
                                         + "load %.2fs, resident-before %.0f MB, stop=%@ %@",
                                         r.variant.tag, Double(r.paramCount) / 1e9,
                                         Double(r.weightsBytes) / 1e6, r.tokens, r.tokensPerSec,
                                         r.loadSeconds, Double(r.residentBytesBeforeLoad) / 1e6,
                                         r.stopReason.rawValue, tags.joined(separator: " ")))
                        self?.log("      \(r.text.debugDescription)")
                        self?.compareResponses[r.variant.tag] = r.text
                    }
                })
            // Match detection needs every response, so it runs once at the end.
            let eligible = compareResponses
            let matched = ResponseAnalysis.matchingTags(eligible)
            log("  match set: \(matched.isEmpty ? "none" : matched.sorted().joined(separator: ", "))")
            log("### compare complete")
        } else if env["LLAMABENCH_MODE"] == "bench" {
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
        setIdleTimerDisabled(true)
        defer {
            isRunning = false
            setIdleTimerDisabled(false)
        }

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

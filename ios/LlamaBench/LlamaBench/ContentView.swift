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

    /// Runs a smoke test without UI interaction when LLAMABENCH_AUTORUN=1.
    ///
    /// Overrides: LLAMABENCH_VARIANT (baseline-fp16|ptq-2bit|qat-2bit),
    ///            LLAMABENCH_BACKEND (cpu|metal).
    func autorunIfRequested() async {
        let env = ProcessInfo.processInfo.environment
        guard env["LLAMABENCH_AUTORUN"] == "1" else { return }

        if let v = env["LLAMABENCH_VARIANT"], let parsed = ModelVariant(rawValue: v) {
            selectedVariant = parsed
        }
        if let b = env["LLAMABENCH_BACKEND"] {
            useCPUOnly = (b.lowercased() != "metal")
        }

        log("### autorun (LLAMABENCH_AUTORUN=1)")
        showEnvironment()
        await runSmokeTest()
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
                Button(state.isRunning ? "Running…" : "Smoke test") {
                    Task { await state.runSmokeTest() }
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

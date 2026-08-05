import SwiftUI

@MainActor
final class CompareState: ObservableObject {

    struct Panel {
        enum Phase { case idle, loading, generating, finished, failed }
        var phase: Phase = .idle
        var text = ""
        var tokens = 0
        var tokensPerSec: Double?
        var loadSeconds: Double?
        var error: String?
        var flags = ResponseAnalysis.Flags()
        var paramCount: UInt64?
        var weightsBytes: UInt64?
        var isMatch = false
    }

    @Published var prompt = "What is the capital of France? Answer in one sentence."
    @Published var panels: [String: Panel] = [:]
    @Published var isRunning = false

    /// Response cap.
    ///
    /// A cap is required because not every variant emits an end-of-turn token:
    /// PTQ produces degenerate loops and runs to the context limit unbounded, and
    /// QAT under-emits EOS on longer answers. 256 leaves room for an explanatory
    /// answer while still bounding those cases; fp16 and QAT finish well under it
    /// on short factual prompts, so raising it costs nothing there.
    ///
    /// The ceiling is n_ctx (2048) minus the prompt.
    @Published var maxTokens = 256
    static let maxTokenChoices = [80, 256, 512]

    func panel(_ v: ModelVariant) -> Panel { panels[v.tag] ?? Panel() }

    /// Recomputed as each response lands, so the badge appears on an earlier panel
    /// once a later one turns out to agree with it.
    private func refreshMatches() {
        // Responses flagged as repetition are excluded, so two degenerate loops do
        // not match each other.
        let eligible = panels.filter { !$0.value.text.isEmpty && !$0.value.flags.repetition }
                             .mapValues(\.text)
        let matched = ResponseAnalysis.matchingTags(eligible)
        for (tag, var p) in panels {
            p.isMatch = matched.contains(tag)
            panels[tag] = p
        }
    }

    func compare() async {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isRunning, !text.isEmpty else { return }

        isRunning = true
        // A three-variant pass includes an fp16 load, so it is long enough that
        // the screen locking mid-run would suspend it.
        ScreenWake.setDisabled(true)
        defer {
            isRunning = false
            ScreenWake.setDisabled(false)
        }

        for v in ModelVariant.allCases { panels[v.tag] = Panel() }

        await PromptComparison.run(
            prompt: text,
            maxTokens: maxTokens,
            onPhase: { [weak self] variant, phase in
                Task { @MainActor in
                    guard let self else { return }
                    var p = self.panels[variant.tag] ?? Panel()
                    switch phase {
                    case .loading:            p.phase = .loading
                    case .generating:         p.phase = .generating
                    case .finished:           p.phase = .finished
                    case .failed(let message): p.phase = .failed; p.error = message
                    }
                    self.panels[variant.tag] = p
                }
            },
            onResponse: { [weak self] r in
                Task { @MainActor in
                    guard let self else { return }
                    var p = self.panels[r.variant.tag] ?? Panel()
                    p.text = r.text
                    p.tokens = r.tokens
                    p.tokensPerSec = r.tokensPerSec
                    p.loadSeconds = r.loadSeconds
                    p.paramCount = r.paramCount
                    p.weightsBytes = r.weightsBytes
                    p.flags = ResponseAnalysis.flags(text: r.text, stopReason: r.stopReason)
                    self.panels[r.variant.tag] = p
                    self.refreshMatches()
                }
            })
    }
}

struct CompareView: View {
    @StateObject private var state = CompareState()

    private let samplePrompts = [
        "What is the capital of France? Answer in one sentence.",
        "Write one sentence explaining what photosynthesis is.",
        "What is 15 plus 27?",
        "List three primary colors.",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Enter a prompt and compare all three variants. Models run one at "
                         + "a time, so only one is in memory at a time.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    TextField("Prompt", text: $state.prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .textFieldStyle(.roundedBorder)
                        .disabled(state.isRunning)

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(samplePrompts, id: \.self) { p in
                                Button(p.prefix(22) + "…") { state.prompt = p }
                                    .font(.caption2)
                                    .buttonStyle(.bordered)
                                    .disabled(state.isRunning)
                            }
                        }
                    }

                    HStack(spacing: 8) {
                        Text("Max tokens").font(.caption).foregroundStyle(.secondary)
                        Picker("Max tokens", selection: $state.maxTokens) {
                            ForEach(CompareState.maxTokenChoices, id: \.self) { n in
                                Text("\(n)").tag(n)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(state.isRunning)
                    }

                    Button {
                        Task { await state.compare() }
                    } label: {
                        HStack {
                            if state.isRunning { ProgressView().controlSize(.small) }
                            Text(state.isRunning ? "Comparing…" : "Compare")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(state.isRunning || state.prompt.trimmingCharacters(in: .whitespaces).isEmpty)

                    ForEach(ModelVariant.allCases) { v in
                        PanelCard(variant: v, panel: state.panel(v))
                    }
                }
                .padding()
            }
            .navigationTitle("Compare")
        }
    }
}

private struct PanelCard: View {
    let variant: ModelVariant
    let panel: CompareState.Panel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(variant.displayName).font(.subheadline).bold()
                if let sub = headerDetail {
                    Text("· \(sub)").font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                statusLabel
            }

            if !badges.isEmpty {
                HStack(spacing: 5) {
                    ForEach(badges, id: \.text) { b in
                        Text(b.text)
                            .font(.caption2).bold()
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(b.color.opacity(0.18))
                            .foregroundStyle(b.color)
                            .clipShape(Capsule())
                    }
                }
            }

            switch panel.phase {
            case .idle:
                Text("Not run yet").font(.caption).foregroundStyle(.tertiary)
            case .loading, .generating:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(panel.phase == .loading ? "loading model…" : "generating…")
                        .font(.caption).foregroundStyle(.secondary)
                }
            case .failed:
                Text(panel.error ?? "failed")
                    .font(.caption).foregroundStyle(.red)
            case .finished:
                if panel.text.isEmpty {
                    Text("(no output)").font(.footnote).foregroundStyle(.tertiary)
                } else if panel.flags.repetition {
                    // The repeating unit is shown once in place of the full loop,
                    // with the raw output available below.
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Repeating unit, \(panel.flags.repeatCount)x:")
                            .font(.caption2).foregroundStyle(.secondary)
                        MarkdownText(markdown: repeatingSample)
                        DisclosureGroup("Show raw output") {
                            Text(panel.text)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                        }
                        .font(.caption2)
                    }
                } else {
                    MarkdownText(markdown: panel.text)
                }
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    /// Parameter count and weight size, read from the model this panel ran.
    private var headerDetail: String? {
        guard let p = panel.paramCount, let w = panel.weightsBytes else { return nil }
        return String(format: "%.3fB · %.0f MB", Double(p) / 1e9, Double(w) / 1e6)
    }

    private struct Badge { let text: String; let color: Color }

    private var badges: [Badge] {
        var out: [Badge] = []
        if panel.flags.repetition {
            out.append(Badge(text: "Repetition detected", color: .orange))
        }
        if panel.flags.truncated {
            out.append(Badge(text: "Cut off at token limit", color: .yellow))
        }
        if panel.isMatch {
            out.append(Badge(text: "Match", color: .green))
        }
        return out
    }

    /// First occurrence of the repeating unit, for display in place of the loop.
    private var repeatingSample: String {
        let words = panel.text.split(separator: " ", omittingEmptySubsequences: true)
        let n = max(panel.flags.repeatUnitWords, 1)
        return words.prefix(n).joined(separator: " ")
    }

    @ViewBuilder private var statusLabel: some View {
        if panel.phase == .finished, let tps = panel.tokensPerSec {
            Text(String(format: "%d tok · %.1f tok/s", panel.tokens, tps))
                .font(.caption2).foregroundStyle(.secondary)
        } else if let l = panel.loadSeconds, panel.phase == .generating {
            Text(String(format: "loaded %.1fs", l))
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

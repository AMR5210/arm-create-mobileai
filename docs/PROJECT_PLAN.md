# Project Plan (v2 — corrected): 2-bit QAT for Arm-Optimized On-Device LLM Inference

**Context for the coding agent reading this:** This supersedes any earlier version of this plan. Use this document as the single source of truth. This is a hackathon project for the *Arm Create: AI Optimization Challenge* (**Track 3: Mobile AI** — phones, tablets, and laptops only, no desktops; confirmed against the official Track Details page). Deadline: **August 14, 2026, 4:00pm PDT**. Read this entire document, including Section 0, before writing any code or making any commits.

---

## 0. Repo hygiene & attribution — do this first, before any code or commits

This is a **public repository**. The following rules apply to everything visible publicly (README, commit messages, PR descriptions, any docs linked from the README):

**Git attribution — disable it completely.**
- No `Co-Authored-By: Claude` trailer on any commit.
- No "Generated with Claude Code" line or footer on any commit or PR.
- The agent must not appear as a contributor in git history.
- Configure this before the first commit, in `~/.claude/settings.json` (or a project-level `.claude/settings.json`):
  ```json
  {
    "attribution": {
      "commit": "",
      "pr": ""
    }
  }
  ```
- After the first commit, verify it does not contain any trailer or footer before proceeding. If it does, the setting wasn't applied correctly — fix it before continuing, not after several commits have already gone out.
- Before the first commit, add the following to `.gitignore`:
  ```
  .claude/*
  !.claude/settings.json
  ```
  This ignores the `.claude/` directory by default but explicitly tracks `settings.json` (project policy, not personal config, no secrets in it), so any other local Claude Code state stays untracked. Run `git status` and confirm only `settings.json` is staged from that directory before committing.

**No personal information in public-facing files.**
- Do not narrate specific personally-owned hardware in the README (e.g. do not write "my Mac mini" or "my iPhone"). Where a device class matters for reproducibility (which chip a benchmark ran on), state it as a plain hardware fact — e.g. "benchmarked on Apple M4" or "benchmarked on an iPhone with the A19 Pro chip" — not framed as personal belongings.
- No other personal details (names, contact info, unrelated personal context) belong in the repo at all.

**README.md must be a clean, public-facing document only** — project overview, what it does, functionality/output, setup and run instructions, license. It is **not** the place for the full internal plan, phase timelines, or hackathon strategy notes.

**This full project plan is an internal reference file for the agent**, not a public artifact. Save it at `docs/PROJECT_PLAN.md` (or `.agent/PROJECT_PLAN.md`) and do not link it from or fold it into the README.

---

## 1. One-line goal

Build and benchmark a small LLM quantized to 2-bit precision using Quantization-Aware Training (QAT), and prove — with real numbers on real Arm hardware — that QAT preserves far more output quality than standard 2-bit Post-Training Quantization (PTQ), at the same memory footprint and inference speed.

## 2. Why this project

- Apple's on-device foundation model (arXiv:2507.13575) uses 2-bit quantization-aware training as a core technique to shrink its 3B-parameter model for Apple Silicon (Arm-based).
- Google shipped official QAT checkpoints for Gemma 4 in 2026, quantizing generation layers to 2 bits and shrinking a 9.6GB model to roughly 1GB.
- Recent 2026 research shows plain 2-bit PTQ quality falls off a cliff, while QAT recovers most of that loss. That gap is exactly what this project measures on Arm hardware.
- This targets the challenge's "Model size," "Model speed," and "Arm-specific optimization" scoring categories with a genuinely reproducible contribution.

## 3. The core experiment

Not just "quantize a model" — this isolates two distinct claims:

- **Claim A:** 2-bit (any method) vs. fp16 → big memory + speed win.
- **Claim B:** 2-bit QAT vs. 2-bit PTQ (same bit-width) → quality win, **not** a speed/memory win — those stay flat by construction.

Three models, benchmarked side by side:

1. **Baseline** — fp16, unmodified.
2. **PTQ-2bit** — same model, quantized post-hoc.
3. **QAT-2bit** — same model, fine-tuned with quantization simulated in the forward pass, exported to the same runtime format as #2.

**Critical constraint:** models 2 and 3 must run through the identical runtime format/kernel (GGUF Q2_K via llama.cpp) so speed/memory differences between them are zero by construction. Only quality (perplexity / instruction-following accuracy) should differ.

## 4. Model & dataset

- **Base model:** Qwen3-0.6B primary. Add Llama-3.2-1B-Instruct only if time allows after the first model works end-to-end.
- **QAT fine-tuning data:** a small open instruction dataset (a few thousand examples). Keep fine-tunes short — hours, not days.
- **Evaluation:** wikitext2 perplexity plus a small instruction-following eval slice (e.g. a subset of MMLU or a simple QA set).

## 5. Toolchain & hardware (corrected)

- PyTorch + Hugging Face `transformers` for QAT fine-tuning.
- Adapt LLM-QAT or EfficientQAT's public code for the fake-quantization forward pass + straight-through estimator, rather than building from scratch.
- Progressive quantization recommended: FP16 → INT4 (PTQ) → INT2 (QAT fine-tune from the INT4 checkpoint), not straight fp16 → INT2.
- llama.cpp for fp16 inference, Q2_K PTQ conversion, and exporting QAT weights into GGUF Q2_K.
- Build llama.cpp with Arm's KleidiAI library enabled (`-DGGML_CPU_KLEIDIAI=ON`), which provides optimized matmul microkernels using SME/SME2/i8mm/dotprod. This is one of Track 3's explicitly named technologies. Note: KleidiAI's public microkernels target int4/int8, not int2 — it will accelerate the intermediate INT4 progressive-quantization checkpoint, but likely not the final Q2_K path directly. State clearly in the write-up which Arm acceleration applies at which precision level; that precision is itself a point in the submission's favor.
- Arm Performix for the official benchmark numbers reported in the submission.

**Hardware roles — these are fixed, do not deviate:**
- **Mac mini M4** — development and iteration only. Build, debug, and validate the pipeline here since there's no cross-compiling friction. **Not** a reported benchmark device.
- **iPhone 17 Pro Max** — primary benchmark device. All final reported numbers and demo video footage come from here, since the Mobile AI track requires phones/tablets/laptops, not desktops.
- **iPhone 12** — secondary device, for an optional cross-generation comparison (older Arm chip vs. newer).
- **GPU cloud instance** — QAT fine-tuning (Phase 3) runs here. Not in this coding sandbox, not on the Mac mini, not on the iPhones.

## 6. Phase-by-phase plan

Today: July 24, 2026. Deadline: Aug 14, 2026, 4:00pm PDT (~3 weeks).

### Phase 0 — Setup (Days 1–2)
- Set up Python environment, PyTorch, `transformers`.
- Build llama.cpp on the Mac mini M4 (native Arm, fastest iteration loop) and verify fp16 inference works end-to-end.
- Download Qwen3-0.6B.
- Set up wikitext2 perplexity script and the small instruction-eval set.

### Phase 1 — Baseline benchmarks (Days 3–4)
- Develop and debug on the Mac mini.
- **Final reported fp16 numbers** (disk size, RAM, tokens/sec, perplexity, instruction-eval) must come from the iPhone 17 Pro Max, not the Mac mini.

### Phase 2 — PTQ 2-bit baseline (Days 5–6)
- Quantize to Q2_K via llama.cpp's built-in tool.
- Same benchmark suite, same device rule: dev on Mac mini, final numbers on iPhone 17 Pro Max.
- Expect a sharp quality drop here — that's the "naive 2-bit" story QAT is meant to fix.

### Phase 3 — QAT implementation (Days 7–15)
- Runs on the GPU cloud instance, not locally.
- Implement fake-quantization + straight-through estimator, adapting LLM-QAT/EfficientQAT code.
- Progressive route: fine-tune from the INT4 PTQ checkpoint down to INT2.
- Short fine-tune on the small instruction dataset — check perplexity trend before extending training time.

### Phase 4 — Export & final benchmarks (Days 16–18)
- Convert QAT-trained weights into GGUF Q2_K.
- Cross-compile llama.cpp for iOS; build a minimal on-device harness (app or CLI) that can run and log benchmarks on iPhone.
- Validate the pipeline on the Mac mini first (fast loop), then port to iPhone 17 Pro Max for the numbers that go in the submission. Run the same suite on iPhone 12 if time allows, for the cross-generation comparison.
- Sanity check: tokens/sec and RAM between PTQ-2bit and QAT-2bit should be nearly identical by design; perplexity/eval score should be the visible differentiator. If speed/memory diverge significantly, debug the export step before proceeding.

### Phase 5 — Polish & submission (Days 19–21)
- Get official Arm Performix numbers from the iPhone 17 Pro Max runs.
- Write the **README** per Section 0's rules: professional, public-facing, no personal hardware framing, no full internal plan.
- Record a demo video under 3 minutes showing the project running on the iPhone (per track requirement), the benchmark table, and a short "same prompt, three models" comparison.
- Capture clear screenshots of the benchmark suite actually running on-device (iPhone 17 Pro Max, and iPhone 12 if that comparison is included) — Track 3's submission format leans on proof artifacts (links/screenshots) since judges can't run a mobile app on someone else's phone themselves. Include these in the submission alongside the repo and video, not as an afterthought.
- Push the repo public with an MIT or Apache 2.0 license visible in the About section. Confirm no commit contains AI attribution trailers.
- Submit before Aug 14, 4:00pm PDT.

## 7. Definition of done

A public repo containing:
- Scripts to reproduce all three models (baseline, PTQ, QAT).
- The benchmark suite and results table/chart.
- Setup instructions to run/validate on iPhone (and Mac mini for dev).
- A clean, professional README with no personal information and no full internal plan.
- This internal plan saved separately (e.g. `docs/PROJECT_PLAN.md`), not linked from the README.
- Git history with no Claude attribution trailers or footers.

## 8. Judging criteria — priority order if time runs short

1. **Technological Implementation (40 pts)** — QAT implementation + Arm kernel integration. Don't cut this.
2. **"WOW" factor (25 pts)** — the side-by-side quality demo on iPhone. Don't cut the video.
3. **Potential Impact (20 pts)** — reusable QAT scripts and benchmark harness.
4. **UX/DX (15 pts)** — setup instructions. Do last, but don't skip.

## 9. Risks & fallbacks

- **QAT fine-tuning doesn't converge in time** → fall back to INT4 QAT vs. INT4 PTQ instead of INT2.
- **iOS cross-compile eats too much time** → this is a Mobile AI track submission, so on-device iPhone benchmarks are not optional; cut the second model (Llama-3.2-1B) or the iPhone 12 comparison before cutting the iPhone 17 Pro Max benchmarks.
- **Arm kernel gives unexpectedly poor speed** → report and explain honestly; still a real finding.

## 10. References to read before coding

- Apple, "Apple Intelligence Foundation Language Models" tech report — arXiv:2507.13575
- LLM-QAT (Liu et al., 2023) — data-free QAT distillation, public code
- EfficientQAT (Chen et al., 2024) — staged optimization QAT
- Recent (2026) research on progressive 2-bit quantization — INT4→INT2 staging, distribution-divergence loss for the QAT objective

## 11. Known variables to disclose honestly in the write-up

- **KleidiAI ISA coverage differs across benchmark devices.** KleidiAI's accelerated matmul kernels depend on newer Arm ISA features (i8mm, dotprod, SME/SME2). The iPhone 17 Pro Max (A19 Pro) is expected to support these; the iPhone 12 (A14) is not. If the iPhone 12 cross-generation comparison is included, call out that any speed delta reflects both chip generation *and* differing ISA/kernel-acceleration support — not a clean apples-to-apples comparison. State which acceleration path applies at which precision level and on which device explicitly, rather than presenting a single clean number.

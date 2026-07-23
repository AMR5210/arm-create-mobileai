# iOS on-device benchmark harness — implementation spec

**For the Claude Code session (or you) implementing this on the Mac mini, with
real Xcode/Swift compiler feedback available.** This document was written in a
sandbox with no macOS, no Xcode, and no iOS device, so it deliberately
contains no Swift source — writing untested Swift against a C API whose
function names have been renamed upstream more than once would create more
debugging work than it saves. What follows is the precise contract the app
must satisfy, so it can be built correctly against whatever the real
`llama.h`/`llama.cpp` C API looks like at build time, verified by an actual
compiler.

## 1. Goal

A minimal iOS app (not a bare CLI — iOS doesn't support that in a way you can
just run standalone on-device) that, for each of the three GGUF model
variants, measures the same five metrics the desktop harness measures
(`scripts/benchmark.py`), and writes one JSON file per variant in the exact
schema `scripts/summarize_results.py` already consumes — so the on-device
results drop into the existing pipeline with zero changes to the Python side.

## 2. Starting point

Fork/extend llama.cpp's own official example app:
```
third_party/llama.cpp/examples/llama.swiftui
```
This is the maintained reference for llama.cpp + Swift + iOS integration — it
already wires up the Swift package, target dependencies, and a working
model-loading/inference path against the current C API. Building a fresh
Xcode project and hand-linking `llama.xcframework` (produced by
`scripts/build_llama_ios.sh`) is more error-prone than starting from this
example and adding a benchmark module to it. Check its current source
(`llama.cpp.swift/LibLlama.swift` or equivalent — the exact file name may have
changed) for the current, correct wrapper around `llama_model_load`,
`llama_init_from_model` / `llama_new_context_with_model` (naming has changed
across llama.cpp versions — use whatever the checked-out version actually
exports), tokenization, batch decode, and logits retrieval. Do not assume any
specific function name from this doc; verify against the actual header.

## 3. Model files to bundle

Add these three `.gguf` files (produced by the desktop pipeline) to the app
target as bundled resources:
- `qwen3-0.6b-fp16.gguf` → tag `baseline-fp16`
- `qwen3-0.6b-ptq-q2_k.gguf` → tag `ptq-2bit`
- `qwen3-0.6b-qat-q2_k.gguf` → tag `qat-2bit`

Also bundle the two eval data files produced by `scripts/prepare_eval_data.py`:
- `eval/data/wikitext2_test.txt`
- `eval/data/instruction_eval.jsonl`

so the on-device perplexity and instruction-eval numbers are computed against
the identical corpora used on desktop — required for the numbers to be
comparable at all.

## 4. Per-variant metrics to measure

Run each of the three bundled models through the same five measurements,
matching the desktop methodology so the comparison is apples-to-apples:

1. **Disk size** — size in bytes of the bundled `.gguf` file
   (`FileManager.attributesOfItem(atPath:)[.size]`).

2. **Peak RAM** — sample process resident memory periodically (e.g. every
   100–200ms on a background timer) for the duration of the benchmark run,
   keep the maximum. Standard iOS technique: `task_info` with the
   `MACH_TASK_BASIC_INFO` flavor, reading `resident_size` from the returned
   `mach_task_basic_info` struct. Reset/restart this sampler for each model
   variant so peak RAM isn't contaminated by a previous run's allocations —
   fully unload the previous model/context before loading the next.

3. **Prompt-processing and generation throughput (tokens/sec)** — mirror
   `llama-bench`'s default methodology so results are comparable to the
   desktop numbers: a prompt-processing pass over a 512-token prompt (measure
   wall-clock around the decode calls that consume it), and a generation pass
   producing 128 tokens (measure wall-clock around the decode calls that
   produce them). Report both `prompt_tokens_per_sec` and `gen_tokens_per_sec`
   separately, same as the desktop JSON schema. Use greedy/temperature-0
   sampling for determinism. Reset the KV cache between the pp and tg passes
   (and between model variants).

4. **Perplexity (WikiText-2)** — same sliding-window formula as
   `qat/eval_utils.py`'s `compute_perplexity`: tokenize the bundled
   `wikitext2_test.txt`, slide a window (suggested `max_length=1024`,
   `stride=512`) over it, at each window compute the mean per-token
   cross-entropy over the *new* (non-overlapping) tokens in that window using
   the model's logits, weight each window's loss by its token count, sum,
   divide by total token count, exponentiate. This requires per-token
   log-probabilities from `llama_get_logits`/`llama_get_logits_ith` for the
   ground-truth next token at each position — check the current API for the
   exact accessor. This is real compute over a real corpus, not a shortcut;
   expect it to take some time on-device, and consider it acceptable to use
   only a prefix of the corpus (documented, consistent across all 3 variants)
   if full-corpus runtime is impractical on a phone.

5. **Instruction-following accuracy** — for each row in
   `instruction_eval.jsonl` (fields: `subject`, `question`, `choices`,
   `answer` — `answer` is a 0-indexed integer into `choices`), build the exact
   same prompt template as `scripts/instruction_eval.py`'s `build_prompt`:
   ```
   Answer the following multiple-choice question with only the letter of the correct choice.

   Question: {question}
   A. {choices[0]}
   B. {choices[1]}
   C. {choices[2]}
   D. {choices[3]}
   Answer:
   ```
   Generate up to 8 tokens at temperature 0, extract the first standalone
   letter A–D from the output (same regex intent as the Python side:
   `\b([ABCD])\b`), compare its index to `answer`. Report overall accuracy and
   optionally per-subject accuracy (`per_subject_accuracy: {subject: float}`),
   matching `instruction_eval.py`'s `run_instruction_eval` return shape.

## 5. Output JSON schema — must match exactly

One JSON file per variant, written with these exact keys (matches
`scripts/benchmark.py`'s `record` dict verbatim, since
`scripts/summarize_results.py` reads these keys directly and does no
key-remapping):

```json
{
  "tag": "baseline-fp16",
  "device": "iPhone 17 Pro Max",
  "model_path": "qwen3-0.6b-fp16.gguf",
  "timestamp_utc": "2026-08-01T12:00:00+00:00",
  "disk_bytes": 1234567890,
  "peak_ram_bytes": 1234567890,
  "prompt_tokens_per_sec": 950.2,
  "gen_tokens_per_sec": 42.7,
  "perplexity": 11.05,
  "instruction_accuracy": 0.62,
  "instruction_per_subject_accuracy": {"high_school_mathematics": 0.6},
  "llama_bench_raw": null
}
```

`tag` must be exactly one of `baseline-fp16`, `ptq-2bit`, `qat-2bit` — these
are the filenames `summarize_results.py` expects (`results/<tag>.json`).
`device` should be the real device model (e.g. `"iPhone 17 Pro Max"`, not a
placeholder), since that string ends up in the public results table.
`llama_bench_raw` is not read by `summarize_results.py` — fine to omit or
leave `null` for on-device results; it exists only to retain llama-bench's
raw desktop output for reference.

## 6. Getting results off the device

There's no filesystem access from a Mac to a sandboxed app's container by
default. Use one of:
- Enable `UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace` in
  Info.plist and write the JSON files to the app's Documents directory — they
  then show up under Files app → On My iPhone → (app name), draggable to a
  Mac or AirDrop-able.
- A `UIActivityViewController` share sheet on a "export results" button.

Whichever mechanism, the end state needed is: the three JSON files copied into
this repo at `results/baseline-fp16.json`, `results/ptq-2bit.json`,
`results/qat-2bit.json`, after which `python scripts/summarize_results.py`
works unmodified.

## 7. Methodology notes

- **Thermal throttling**: mobile SoCs throttle under sustained load. Don't run
  all three variants back-to-back with no gap — let the device cool between
  runs, or at minimum note in the write-up if throttling was observed (a
  tokens/sec number that drops mid-run is a sign of it).
- **Determinism**: use greedy/temperature-0 sampling everywhere a metric is
  reported, matching the desktop scripts, so re-runs are comparable.
- **Screenshots**: Track 3's submission format wants proof artifacts since
  judges can't run a mobile app on someone else's phone. Capture the app
  mid-run and showing the final per-variant numbers, on both iPhone 17 Pro
  Max and (if time allows) iPhone 12.
- **KleidiAI**: if the xcframework build in `scripts/build_llama_ios.sh` picks
  up KleidiAI-accelerated kernels for iOS/Arm, note in the write-up that its
  microkernels target int4/int8, not int2 — it may accelerate an INT4 path
  but not directly the final Q2_K comparison. Confirm what the actual iOS
  build enables rather than assuming parity with the Mac mini build flags in
  `scripts/build_llama_cpp.sh`.

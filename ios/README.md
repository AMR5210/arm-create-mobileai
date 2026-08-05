# iOS on-device benchmark harness

Runs the three GGUF model variants (fp16 baseline, PTQ 2-bit, QAT 2-bit) on
Arm-based iOS hardware and reports the same metrics as the desktop harness
(`scripts/benchmark.py`).

All five metrics are implemented and have been run on an iPhone 17 Pro Max;
`results/<tag>.json` holds those figures. See
[Running the full suite](#3-running-the-full-suite) to reproduce.

## Prerequisites

- macOS with Xcode installed (developed against Xcode 26.6).
- `third_party/llama.cpp` cloned — `scripts/build_llama_cpp.sh` does this.
- The three `.gguf` files in `models/`, produced by the desktop pipeline.

## 1. Build the framework

```bash
./ios/build_llama_xcframework.sh
```

This cross-compiles llama.cpp into `ios/Frameworks/llama.xcframework` (iOS
device + simulator, arm64) with Arm KleidiAI microkernels enabled, and confirms
the kernels are present in the output.

The script wraps llama.cpp's own `build-xcframework.sh` rather than calling it
directly, because that script hardcodes its cmake flags and three of its defaults
do not suit this build:

| Upstream default | Effect here | Adjustment |
| --- | --- | --- |
| No `-DGGML_CPU_KLEIDIAI` | KleidiAI not compiled in | flag injected |
| `GGML_NATIVE=OFF` with no `GGML_CPU_ARM_ARCH` | KleidiAI compiles without microkernels | `GGML_CPU_ARM_ARCH` set explicitly |
| Simulator slice built `arm64;x86_64` | an ARM `-march` is invalid for the x86_64 compile | simulator restricted to arm64 |

The second row warrants detail. `ggml-cpu/CMakeLists.txt` runs its `-march`
feature detection only when `GGML_NATIVE=ON`. Cross-compiling requires it `OFF`,
since the target CPU cannot be probed from the host, and in that configuration
`ARCH_FLAGS` receives no `-march` unless `GGML_CPU_ARM_ARCH` is supplied. The
KleidiAI block inspects `ARCH_FLAGS` for `+dotprod`/`+i8mm`/`+sme` to select
microkernels, so without it the pack helpers build with no matmul kernels:
`ctx.kernels_*` remain null and `supports_op()` returns false for every tensor.
The build still succeeds and still logs "Using KleidiAI optimized kernels if
applicable", so setting `GGML_CPU_ARM_ARCH` is what makes the flag effective in
practice.

`verify_kleidiai.sh` checks for `kai_run_matmul*` symbols in the built binary,
which confirms the microkernels are present independently of build success.

Current output:

```
--- ios-arm64
    archs:      arm64
    kai_run_matmul total:    16   (dotprod=12 i8mm=4 sme=0)
    RESULT: OK - KleidiAI microkernels present.
```

### Choosing the Arm baseline

`ARM_ARCH` is a parameter because it is compiled into every ggml CPU source: a
binary built for a newer baseline raises SIGILL on an older chip.

| `ARM_ARCH` | Runs on | KleidiAI kernels |
| --- | --- | --- |
| `armv8.6-a+dotprod+i8mm+fp16` (default) | A15 and newer | dotprod + i8mm |
| `armv8.2-a+dotprod+fp16` | A14 and newer | dotprod only |

The default targets the primary benchmark device. For an A14 (iPhone 12)
cross-generation run, build a second framework:

```bash
ARM_ARCH=armv8.2-a+dotprod+fp16 ./ios/build_llama_xcframework.sh
```

SME is not enabled: `HAVE_SME` fails for this target, and llama.cpp logs
`kleidiai: SME disabled` at runtime.

## 2. Build and run the app

```bash
cd ios/LlamaBench
xcodebuild -project LlamaBench.xcodeproj -scheme LlamaBench -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro Max' \
  -derivedDataPath build/DerivedData build
```

The app finds models in three places, in priority order:

1. **App bundle** — how models ship for a real device run.
2. **Documents directory** — files pushed via Files.app, avoiding a 2.4 GB app
   bundle during iteration. `UIFileSharingEnabled` is set.
3. **`LLAMABENCH_MODEL_DIR`** — a host directory, readable from the simulator.

The environment variable keeps local absolute paths out of the repository.

### Headless run

The app runs without UI interaction, making benchmark runs scriptable:

```bash
SIMCTL_CHILD_LLAMABENCH_AUTORUN=1 \
SIMCTL_CHILD_LLAMABENCH_VARIANT=ptq-2bit \
SIMCTL_CHILD_LLAMABENCH_BACKEND=cpu \
SIMCTL_CHILD_LLAMABENCH_MODEL_DIR="$PWD/../../models" \
  xcrun simctl launch --console-pty <device-udid> org.armcreate.llamabench
```

`LLAMABENCH_VARIANT` is `baseline-fp16` | `ptq-2bit` | `qat-2bit`;
`LLAMABENCH_BACKEND` is `cpu` | `metal`.

## Backend selection

llama.cpp offloads to Metal by default on iOS. With Metal running the matmuls,
the CPU backend — and therefore KleidiAI — does not execute. Any figure reported
as an Arm-CPU or KleidiAI result must come from the `cpuOnly` backend
(`n_gpu_layers = 0`). The `Backend` enum in `LlamaRunner.swift` makes this an
explicit choice at every call site rather than an inherited default.

## KleidiAI coverage by quantization type

From llama.cpp's runtime log, captured in-app (`LlamaLog` hooks `llama_log_set`,
since a device provides no stderr to read):

```
kleidiai: primary q4 kernel feature DOTPROD
kleidiai: primary q8 kernel feature I8MM
kleidiai: no compatible f32 kernels found for CPU features mask 3
kleidiai: SME disabled
```

KleidiAI's `supports_op()` in this llama.cpp revision accepts `GGML_TYPE_Q4_0`,
`GGML_TYPE_Q8_0`, `GGML_TYPE_F32`, and one `GGML_TYPE_F16` path. `Q2_K` is not
among them, so both 2-bit variants run their weight matmuls on stock ggml CPU
kernels with KleidiAI loaded but not dispatched.

docs/METHODOLOGY.md records this direction ("KleidiAI's public microkernels
target int4/int8, not int2"). The measured refinement is that the effect on the
Q2_K path is zero rather than reduced, and the write-up should state that
directly.

## Inference verification

All three variants load and generate on an arm64 iOS target (iPhone 17 Pro Max
simulator, CPU backend, greedy/temperature-0, prompt `"The capital of France is"`).
Run `scripts/verify_model_signatures.py` first to confirm model identity.

| Variant | Disk | Weights | Params | Load | 16-token gen | Output |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline-fp16` | 1509.3 MB | 1503.4 MB | 0.752 B | 1.25 s | 39.5 tok/s | `Paris. The capital of Italy is Rome. The capital of Spain is Madrid.` |
| `ptq-2bit` | 479.8 MB | 473.8 MB | 0.596 B | 0.25 s | 65.1 tok/s | `the most important factor in the management of the business in the case of which is` |
| `qat-2bit` | 495.2 MB | 489.2 MB | 0.596 B | 0.19 s | 58.2 tok/s | `Paris, and the capital of Spain is Madrid. The capital of Italy is Rome` |

The fp16 row serves as the control: coherent output there establishes that
degraded 2-bit output is a property of the models rather than of the harness.

This 16-token raw-completion probe is a liveness check, not a quality measure —
the prompt is a bare completion prefix rather than the model's chat format, and
16 tokens is short for a reasoning model. See
[docs/GENERATION_QUALITY.md](../docs/GENERATION_QUALITY.md) for the quality
comparison, which uses the Qwen3 chat template.

These are simulator figures on desktop-class silicon, included to demonstrate the
harness rather than as reportable results. Per docs/METHODOLOGY.md, reported
numbers come from physical iPhone hardware.

### Footprint parity between the two 2-bit variants

Claim B (docs/METHODOLOGY.md) requires tokens/sec and RAM between `ptq-2bit` and
`qat-2bit` to be closely matched, so that quality is the variable under test.

Parity is established in `scripts/quantize_ptq.sh`: the PTQ baseline is quantized
with `--token-embedding-type f16 --output-tensor-type f16`, then has its duplicate
`output.weight` dropped so its embeddings are tied like the QAT export. Both
variants carry the identical F16 embedding, so no embedding-precision advantage is
attributed to QAT training.

| | `ptq-2bit` | `qat-2bit` |
| --- | --- | --- |
| Tensors | 310 | 310 |
| Elements | 0.5960 B | 0.5960 B |
| Tensor names | identical sets | identical sets |
| `token_embd.weight` | F16 (311.2 MB) | F16 (311.2 MB) |
| `output.weight` | absent (tied) | absent (tied) |
| Disk | 479.8 MB | 495.2 MB |

Remaining divergence is confined to the transformer body and runs in both
directions:

| Relationship | Count | Detail |
| --- | --- | --- |
| Identical type | 218 | — |
| PTQ higher precision | 83 | PTQ `Q3_K` vs QAT `Q2_K` |
| QAT higher precision | 9 | QAT `F16` skip-layers vs PTQ `Q2_K`/`Q3_K` |

Excluding the shared 311.2 MB embedding, the transformer body is 168.6 MB for PTQ
against 184.0 MB for QAT — approximately 9% larger, since QAT's 9 fake-quant
skip-layers are held at F16. Disk, RAM and speed are therefore closely matched
rather than identical by construction.

Exact precision parity is available if wanted: quantizing PTQ with `--pure` Q2_K
plus `--tensor-type` F16 on the same 9 skip-layer tensors would leave training as
the only difference between the variants. This lowers PTQ's precision on 83
tensors, so it is an experiment-design choice rather than a defect to correct.

## 3. Running the full suite

```bash
DEVICE_ID=<devicectl-device-id> ./ios/run_benchmark_suite_device.sh
```

The driver runs one variant per app process, detects completion from the app's
log, and retrieves each record when the run finishes. `ios/run_benchmark_suite.sh`
is the simulator equivalent.

Inputs are staged into the app's data container before the run, and records are
pulled back afterwards:

```bash
# stage a model (repeat for each variant, plus eval/data/*)
xcrun devicectl device copy to --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier org.armcreate.llamabench \
  --source models/qwen3-0.6b-qat-q2_k.gguf \
  --destination Documents/qwen3-0.6b-qat-q2_k.gguf

# retrieve a record
xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier org.armcreate.llamabench \
  --source Documents/results/qat-2bit.json --destination results/qat-2bit.json
```

`UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` are set, so the
same files are reachable through Files.app → On My iPhone → LlamaBench if a
cable-free path is wanted.

### Record schema

One record per variant at `results/<tag>.json`, where `tag` is one of
`baseline-fp16`, `ptq-2bit`, `qat-2bit` — the filenames
`scripts/summarize_results.py` reads. The first twelve keys match
`scripts/benchmark.py`'s `record` dict exactly, including explicit `null`s, so a
desktop record and a device record have the same shape:

```json
{
  "tag": "qat-2bit",
  "device": "iPhone 17 Pro Max (iPhone18,2)",
  "model_path": "qwen3-0.6b-qat-q2_k.gguf",
  "timestamp_utc": "2026-08-05T08:12:30Z",
  "disk_bytes": 495193952,
  "peak_ram_bytes": 2181038080,
  "prompt_tokens_per_sec": 758.27,
  "gen_tokens_per_sec": 67.30,
  "perplexity": 18.46,
  "instruction_forced_choice_accuracy": 0.18,
  "instruction_per_subject_forced_choice_accuracy": { "...": 0.2 },
  "llama_bench_raw": null,

  "model_sha256": "a861b892...",
  "backend": "CPU (Arm/KleidiAI eligible)",
  "harness": { "perplexity_method": "...", "throughput_clock_settling_note": "..." }
}
```

`llama_bench_raw` is a desktop-only field and is always `null` on device.
`model_sha256`, `backend` and `harness` are additions beyond the shared schema;
`summarize_results.py` ignores unknown keys. `harness` carries the methodology
notes for each metric, so a record explains how its own figures were produced.

`device` names the specific model rather than the device family, derived from
`hw.machine` — see `DeviceInfo.swift`.

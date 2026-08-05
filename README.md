# 2-bit QAT for on-device LLM inference on Arm

Quantization-aware training takes Qwen3-0.6B to 2 bits and recovers **12× lower
perplexity than post-training quantization at the same footprint**, measured on an
iPhone 17 Pro Max.

Built for the Arm Create: Mobile AI Challenge (Track 3).

| Variant | Disk | Peak RAM | Prompt tok/s | Gen tok/s | WikiText-2 ppl |
|---|---|---|---|---|---|
| fp16 baseline | 1509.3 MB | 4220.6 MB | 819.23 | 47.34 | 21.37 |
| PTQ 2-bit | 479.8 MB | 2149.3 MB | 686.39 | 64.27 | 220.91 |
| **QAT 2-bit** | **495.2 MB** | **2181.0 MB** | **758.27** | **67.30** | **18.46** |

iPhone 17 Pro Max (A19 Pro), CPU backend, 4 threads. Full records with model
hashes in [`results/`](results/); regenerate the table with
`python scripts/summarize_results.py`.

The on-device perplexity figures reproduce llama.cpp's own `llama-perplexity` to
within **0.011%** on all three variants, which is what makes the comparison worth
reading.

<!-- SCREENSHOT GOES HERE: Compare tab on device, France prompt — fp16 and QAT
     showing the Match badge with identical answers, PTQ showing Repetition
     detected. -->

**Built with** PyTorch · Hugging Face Transformers · llama.cpp / GGUF · Arm
KleidiAI · Swift / SwiftUI · iOS

---

## What this measures

Two claims, measured separately (full definitions in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)):

- **Claim A — 2-bit vs fp16.** A size and speed win: 3.1× smaller on disk, 1.9×
  less RAM, 1.4× faster generation. Either quantization method delivers this, so
  it says nothing about QAT specifically.
- **Claim B — 2-bit QAT vs 2-bit PTQ.** A quality win at the same bit width. Both
  variants run the same GGUF `Q2_K` format through the same llama.cpp kernels, so
  quality is meant to be the only variable.

Claim B is the one that is easy to get wrong. A 2-bit variant that quietly keeps
more tensors at higher precision wins on quality without QAT contributing
anything, so the two are held to matched structure: 310 tensors, 0.5960 B
elements, identical tensor names, both with F16 tied embeddings. The residual
difference is 3.2% on disk and 1.5% on RAM, from 9 fake-quant skip-layers the QAT
export holds at F16. That is stated rather than described as identical.

Off-domain check: on C4, which contributes nothing to the training blend, QAT
measures 32.25 against PTQ's 279.30 — an 8.7× gap rather than the 12.0× seen on
WikiText-2. Part of the WikiText-2 margin is domain alignment, and
[`results/c4_perplexity.json`](results/c4_perplexity.json) exists to catch that.

## Quickstart

Requires Python 3.10+, `cmake`, and a C toolchain. QAT training additionally needs
a GPU and several hours; everything else runs on a laptop.

```bash
./scripts/setup_env.sh                 # venv + dependencies
source .venv/bin/activate
./scripts/build_llama_cpp.sh           # clones and builds llama.cpp
python scripts/download_model.py       # base model -> models/qwen3-0.6b-hf
./scripts/convert_to_gguf.sh           # -> models/qwen3-0.6b-fp16.gguf
./scripts/quantize_ptq.sh              # -> models/qwen3-0.6b-ptq-q2_k.gguf
python scripts/prepare_eval_data.py    # WikiText-2 + C4 eval corpora
```

The QAT checkpoint is published, so the comparison reproduces without retraining:

```bash
huggingface-cli download AMR5210/qwen3-0.6b-qat-q2k \
  fineweb-blend/qwen3-0.6b-qat-fineweb-blend-q2_k.gguf --local-dir models/
python scripts/verify_model_signatures.py    # confirms all three files by SHA-256
```

**See the difference in one command** — the same prompt through all three
variants:

```bash
./scripts/compare_generations.sh
```

Then `python scripts/benchmark.py --tag ptq-2bit --device "<host>" --model
models/qwen3-0.6b-ptq-q2_k.gguf` for the desktop metrics, or
[`ios/README.md`](ios/README.md) for the on-device harness.

To train rather than download, see [`docs/AMD_ROCM_SETUP.md`](docs/AMD_ROCM_SETUP.md)
(or [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md)) and `scripts/train_qat.py --help`.

## Layout

| Path | Contents |
|---|---|
| `qat/` | QAT implementation: fake quantization, Q2_K encoder, distillation losses |
| `scripts/` | Pipeline: setup, convert, quantize, train, evaluate, benchmark |
| `ios/` | iOS benchmark harness and Compare app ([README](ios/README.md)) |
| `results/` | Per-variant records, perplexity analyses, generated summary |
| `docs/` | Methodology, generation-quality findings, training-environment setup |
| `eval/data/` | Generated eval corpora (not committed) |

## How the QAT works

Weights are fake-quantized per group with an affine min/max scheme and a
straight-through estimator, using the full asymmetric integer range — at 2 bits,
where there are four codes in total, spending one on a symmetric zero point is
expensive. Training starts from a Q2_K-rounded checkpoint rather than fp16 and
distills against the fp16 teacher with BitDistiller's CAKLD objective. Nine
outlier-heavy layers are held at full precision.

### The export path is the interesting part

The obvious way to ship a QAT checkpoint as GGUF is to write fp16 weights and run
`llama-quantize`. That silently defeats the point of QAT: `llama-quantize`
**re-derives its own Q2_K parameters** from those fp16 weights, so the model that
runs on device is not the model training optimized.

[`qat/gguf_q2k.py`](qat/gguf_q2k.py) is a pure-numpy Q2_K encoder that removes the
round trip — a port of ggml's reference quantiser (`quantize_row_q2_K_ref` /
`make_qkx2_quants`), including Q2_K's two-level structure: 256-element super-blocks
split into 16 sub-blocks, whose scales and minima are themselves 4-bit-quantized
under an fp16 super-block scale. Training uses `--group-size 16` so the QAT groups
line up with those sub-blocks and trained values land on the deployment grid.

Correctness is checkable without a C toolchain: packed blocks are re-decoded with
gguf-py's trusted `Q2_K.dequantize_blocks`. A wrong byte layout cannot survive that
round trip, which separates "is the packing correct" from "is the quantizer good".

## Measurement

The numbers above are only useful if the harness measures what it claims, so:

- **Perplexity replicates `llama-perplexity` rather than approximating it** —
  non-overlapping 512-token chunks, KV cache cleared per chunk, scoring positions
  `[256, 511)`. On-device results land within 0.011% of the desktop reference on
  every variant.
- **KV clearing is verified, not assumed.** `llama_memory_seq_pos_max` is read
  either side of each throughput rep: `-1` after every clear, `511` after every
  512-token pass. Cache reuse would have inflated later reps.
- **Model identity is checked before measurement.**
  `scripts/verify_model_signatures.py` validates size, tensor mix, skip-layer set
  and SHA-256 against [`scripts/model_signatures.json`](scripts/model_signatures.json).
  Every record embeds the hash of the file it measured.
- **Throughput is the mean of 5 reps**, matching `llama-bench`. Per-rep samples
  decline 15–19% across a run, too fast for thermal throttling — boost-clock
  settling — so the reported mean is conservative against the first-rep peak.

Details in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and
[`docs/GENERATION_QUALITY.md`](docs/GENERATION_QUALITY.md).

## What didn't work

Negative results are recorded with the same detail as positive ones, in
[`results/wikitext2_perplexity.json`](results/wikitext2_perplexity.json) under
`analysis`.

| Tried | Result | Action |
|---|---|---|
| PTQ baseline with Q2_K embeddings | Gave QAT an advantage it had not earned — embeddings were never trained under fake-quant | Rebuilt PTQ with F16 tied embeddings. Made **PTQ 20% better** (275.89 → 220.91); the QAT result now stands against a stronger baseline |
| Sliced-Wasserstein distillation | Lost to CAKLD at matched loss share, 27.40 vs 26.95 | Closed. The first run was also confounded: `--distill-weight` is not comparable across loss types, and 0.5 gave Wasserstein 69–74% of total loss against CAKLD's 10–20% |
| ParetoQ SEQ quantizer | Confidence intervals overlap — a tie, not a win | Reported as a tie. The real finding was init sensitivity: a 1.49× swing from init mode alone |
| Progressive 4→2 bit annealing | No measurable effect | Confirmed with a dedicated control run, then disabled |
| Instruction-following accuracy | Every variant at or below the 25% chance line under three scoring methods | Dropped from reported metrics. Forced-choice scoring showed why: `qat-2bit` picks "A" for all 100 questions, exactly the 18% constant-A rate |
| KleidiAI on the 2-bit path | `supports_op` covers Q4_0/Q8_0/F32, not Q2_K | Stated plainly: zero acceleration on the path the headline result uses |

A stale QAT export also went undetected long enough to produce misleading
measurements — same filename, plausible size, three extra skip-layers, matching no
recorded run. `verify_model_signatures.py` and `model_signatures.json` exist
because of it, and every record now embeds its model's hash.

## Limitations

- **No fp16 ceiling run.** Ratios against unadapted fp16 conflate domain
  adaptation with the cost of quantization. QAT's 18.46 beating fp16's 21.37 on
  WikiText-2 is *not* evidence that 2-bit beats fp16 — fp16 never saw the training
  blend. A fair ceiling was not run.
- **Footprint parity is close, not exact**: 3.2% disk, 1.5% RAM.
- **KleidiAI contributes nothing to the 2-bit path.** Its microkernels target
  int4/int8; `Q2_K` matmuls run on stock ggml CPU kernels.
- **One base model, one device.** Qwen3-0.6B on A19 Pro. An iPhone 12
  cross-generation run needs a second framework build at `armv8.2-a`, since the
  default baseline raises SIGILL on A14.
- **No live demo.** The app needs code signing and 2.4 GB of model files, so the
  screenshot and [`results/`](results/) are the evidence.

## Contributing

Pull requests welcome. For any change touching the models or the export path,
`python scripts/verify_model_signatures.py` should pass before and after — it
checks size, tensor mix, skip-layer set and SHA-256 for all three variants.

## License

MIT — see [LICENSE](LICENSE).

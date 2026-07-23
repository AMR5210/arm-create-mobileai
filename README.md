# arm-create-mobileai

2-bit Quantization-Aware Training (QAT) for on-device LLM inference on Arm hardware.

This project builds and benchmarks three variants of the same small language model:

1. **Baseline (fp16)** — the unmodified model.
2. **PTQ-2bit** — the same model quantized to 2 bits post-hoc.
3. **QAT-2bit** — the same model fine-tuned with 2-bit quantization simulated during training, then exported to the same runtime format as the PTQ variant.

All quantized variants run through the identical inference format ([GGUF](https://github.com/ggml-org/ggml) `Q2_K` via [llama.cpp](https://github.com/ggml-org/llama.cpp)), so speed and memory footprint are the same by construction between PTQ and QAT. The only thing that should differ is output quality — measured with WikiText-2 perplexity and a small instruction-following evaluation slice — which is the actual claim this project measures: **QAT preserves substantially more model quality than PTQ at the same 2-bit size and speed.**

## Status

Work in progress. This README will be updated as each stage (baseline, PTQ, QAT, on-device benchmarking) lands.

## Repository layout

```
scripts/                Setup, build, and data-preparation scripts
eval/                    Evaluation harness and data
  data/                  Generated eval corpora (not committed; see prepare_eval_data.py)
docs/                    Internal project reference docs
results/                 Benchmark output (not committed)
```

## Setup

Requires Python 3.10+, `git`, and `cmake`.

```bash
scripts/setup_env.sh          # creates .venv and installs Python dependencies
source .venv/bin/activate

scripts/build_llama_cpp.sh    # clones + builds llama.cpp (enables Arm KleidiAI kernels on Arm hosts)
scripts/download_model.py     # downloads the base model (Qwen3-0.6B by default)
scripts/prepare_eval_data.py  # prepares WikiText-2 perplexity corpus + instruction-eval slice
```

## Benchmark suite

Every model variant (fp16 baseline, PTQ-2bit, QAT-2bit) is measured with the same harness, so numbers are directly comparable across variants:

```bash
# 1. Convert a Hugging Face checkpoint to fp16 GGUF (the common base for all variants)
scripts/convert_to_gguf.sh models/qwen3-0.6b-hf models/qwen3-0.6b-fp16.gguf

# 2. Run the full benchmark suite: disk size, peak RAM, tokens/sec, perplexity, instruction accuracy
python scripts/benchmark.py \
  --tag baseline-fp16 \
  --device "iPhone 17 Pro Max" \
  --model models/qwen3-0.6b-fp16.gguf

# 3. Combine every results/*.json into one comparison table
python scripts/summarize_results.py
```

`--device` matters: only iPhone-class runs are reported as final numbers; runs on the development machine are for pipeline validation only, not for the submission's benchmark table.

## Hardware notes

- Reported benchmark numbers are collected on real Arm mobile hardware (phone/tablet/laptop class), per the target track's requirements, not on desktop hardware.
- Where KleidiAI-accelerated kernels are used, note that their optimized microkernels target int4/int8 precision; the final 2-bit (`Q2_K`) path does not currently benefit from the same acceleration. Which acceleration path applies at which precision, and on which device's Arm ISA feature set (e.g. i8mm/dotprod/SME2 availability), is stated explicitly alongside the relevant benchmark numbers rather than left implicit.

## License

[MIT](LICENSE)

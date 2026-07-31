# QAT GGUF export — testing runbook

The export pipeline (`scripts/export_qat_gguf.py`) runs anywhere with Python +
the `gguf` package + llama.cpp's pure-Python `convert_hf_to_gguf.py` (no C
build). **Measuring quality** needs the compiled `llama-perplexity`, so run the
perplexity step on a box where llama.cpp is built (the Mac mini / Colab / AMD
droplet), not the Windows dev GPU box.

## What the three fixes do

- **Fix 1 (Q2_K encoder):** 2-bit tensors are packed by our own faithful port
  of ggml's Q2_K reference quantiser (`qat/gguf_q2k.py`), so the scheme is
  controlled in-process instead of black-boxed through `llama-quantize`. Train
  with `--group-size 16` (now the default) so the QAT grouping matches Q2_K's
  16-element sub-blocks.
- **Fix 2 (skip-layers stay F16):** the layers kept full precision during QAT
  (`--skip-layers`) are written as F16, not crushed to Q2_K. Pass the *same*
  `--skip-layers` patterns to the exporter.
- **Fix 3 (tokenizer restore):** the base model's tokenizer files are copied
  into the HF dir before conversion (QAT never trains the tokenizer).

## Local validation already done (Windows box, no llama.cpp)

Produced `models/qwen3-0.6b-qat-q2_k.gguf` (468 MB) from the 500-step checkpoint:
- tensors: **188 Q2_K, 8 F16 (skip-layers), 9 F16 (embeddings), 113 F32 (norms)**;
- every skip-layer confirmed F16 and **bit-identical** to the fp16 source;
- Q2_K tensors read back from disk and dequantised (via gguf-py's reference
  decoder) at **~6.7–7.2 % mean rel error** — normal for Q2_K, and proof the
  byte layout is correct;
- all GGUF metadata + the full 151936-token tokenizer preserved.

Structure is verified; only the perplexity number is pending a llama.cpp box.

## Steps on the llama.cpp box

```bash
# 0. Build llama.cpp (once)
scripts/build_llama_cpp.sh
LLAMA=third_party/llama.cpp/build/bin

# 1. PTQ Q2_K baseline (the ~276 reference), from the ORIGINAL base model
scripts/convert_to_gguf.sh models/qwen3-0.6b-hf models/qwen3-0.6b-fp16.gguf
scripts/quantize_ptq.sh    models/qwen3-0.6b-fp16.gguf models/qwen3-0.6b-ptq-q2_k.gguf Q2_K

# 2. QAT Q2_K export (copy the .gguf produced on the dev box, or re-run the exporter here)
python scripts/export_qat_gguf.py \
  --hf-dir models/qwen3-0.6b-qat-hf --base-model models/qwen3-0.6b-hf \
  --out models/qwen3-0.6b-qat-q2_k.gguf \
  --skip-layers layers.16.self_attn.k_proj layers.21.self_attn.k_proj \
                layers.8.self_attn.k_proj  layers.27.self_attn.k_proj \
                layers.26.mlp.gate_proj    layers.26.mlp.up_proj \
                layers.27.mlp.gate_proj    layers.27.mlp.up_proj

# 3. Perplexity for both, same corpus
$LLAMA/llama-perplexity -m models/qwen3-0.6b-ptq-q2_k.gguf -f eval/data/wikitext2_test.txt
$LLAMA/llama-perplexity -m models/qwen3-0.6b-qat-q2_k.gguf -f eval/data/wikitext2_test.txt
```

## How to read the result — and an honest expectation

Three numbers, three different measurements:

| number | what it is |
|---|---|
| ~276 | PTQ Q2_K of the **original** weights (llama-perplexity, full corpus) |
| ~7,551 | training-time proxy: the fake-quant PyTorch model, **group-32**, truncated corpus |
| (this test) | QAT **Q2_K** via `llama-perplexity`, full corpus |

**Expectation for the current checkpoint:** the QAT number will likely land in
the thousands, i.e. it will *not* beat PTQ's ~276. That is not an export bug —
the exporter is verified correct. It reflects the underlying checkpoint: this
500-step run (batch-1 / Adafactor / **group-32** / aggressive 2-bit on a 6 GB
card) produced a model whose 2-bit quality is genuinely poor (~7,551 proxy).
Export plumbing cannot manufacture quality that is not in the weights.

**What actually closes the gap:** retrain with the export-aligned settings now
in place, then re-export and re-test:

```bash
python scripts/train_qat.py --base-model models/qwen3-0.6b-hf \
  --output-dir models/qwen3-0.6b-qat-hf \
  --group-size 16 \
  --skip-layers layers.16.self_attn.k_proj ... layers.27.mlp.up_proj
# (on a bigger GPU: default AdamW + batch 4; longer than 500 steps)
```

The `--group-size 16` default aligns training with Q2_K sub-blocks so the values
the model learns actually survive Q2_K packing. Expect the QAT-vs-PTQ comparison
to become meaningful only after such a retrain.

## Size caveat

Keeping the 8 skip-layers **and** the embeddings at F16 makes the QAT GGUF
larger than a vanilla PTQ Q2_K (here 468 MB vs a fully-Q2_K file). This is the
deliberate cost of mixed precision; state it alongside the benchmark so the
PTQ-vs-QAT comparison notes the size difference rather than implying identical
footprints.

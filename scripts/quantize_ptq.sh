#!/usr/bin/env bash
# Post-training quantization (PTQ) via llama.cpp's built-in quantizer.
# Produces the "naive 2-bit" baseline used to demonstrate the QAT quality gap
# (Claim B in docs/PROJECT_PLAN.md: PTQ-2bit vs QAT-2bit, same runtime format,
# quality is the only thing that should differ).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANTIZE_BIN="$ROOT_DIR/third_party/llama.cpp/build/bin/llama-quantize"

FP16_GGUF="${1:-models/qwen3-0.6b-fp16.gguf}"
OUT_GGUF="${2:-models/qwen3-0.6b-ptq-q2_k.gguf}"
QUANT_TYPE="${3:-Q2_K}"

if [ ! -x "$QUANTIZE_BIN" ]; then
  echo "ERROR: $QUANTIZE_BIN not found/executable. Run scripts/build_llama_cpp.sh first." >&2
  exit 1
fi

if [ ! -f "$FP16_GGUF" ]; then
  echo "ERROR: $FP16_GGUF not found. Run scripts/convert_to_gguf.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_GGUF")"

# Embeddings are kept at F16 and left tied, matching the QAT exporter's treatment,
# for two reasons:
#
#   1. Comparison fairness. The QAT model's embeddings were never trained under
#      fake-quantization. Quantizing the PTQ model's embeddings to Q2_K while the
#      QAT model keeps F16 would attribute an embedding-precision advantage to
#      QAT training. Both variants carry the identical F16 embedding.
#
#   2. Footprint parity. Claim B in docs/PROJECT_PLAN.md requires the two 2-bit
#      variants to share a footprint so that quality is the variable under test.
#
# `--output-tensor-type f16` alone is insufficient. Qwen3-0.6B sets
# `tie_word_embeddings: true`, yet convert_hf_to_gguf.py materializes
# `output.weight` as a byte-identical duplicate of `token_embd.weight`.
# llama-quantize can retype a tensor but not remove one, so the duplicate is
# dropped in a second pass to leave the model tied like the QAT export. llama.cpp
# falls back to `token_embd.weight` for the output projection when `output.weight`
# is absent, making this numerically lossless: greedy generations are identical
# before and after the drop.
TMP_GGUF="$(dirname "$OUT_GGUF")/.$(basename "$OUT_GGUF").untied.tmp"

echo "==> Quantizing $FP16_GGUF -> $TMP_GGUF ($QUANT_TYPE, F16 embeddings)"
"$QUANTIZE_BIN" \
  --token-embedding-type f16 \
  --output-tensor-type f16 \
  "$FP16_GGUF" "$TMP_GGUF" "$QUANT_TYPE"

echo "==> Dropping duplicate output.weight to leave embeddings tied"
PY_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3
"$PY_BIN" "$ROOT_DIR/scripts/gguf_drop_tensor.py" \
  "$TMP_GGUF" "$OUT_GGUF" \
  --drop output.weight \
  --require-duplicate-of token_embd.weight

rm -f "$TMP_GGUF"

echo "==> Done:"
ls -lh "$OUT_GGUF"

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

echo "==> Quantizing $FP16_GGUF -> $OUT_GGUF ($QUANT_TYPE)"
"$QUANTIZE_BIN" "$FP16_GGUF" "$OUT_GGUF" "$QUANT_TYPE"

echo "==> Done:"
ls -lh "$OUT_GGUF"

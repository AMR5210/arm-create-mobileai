#!/usr/bin/env bash
# Converts a Hugging Face checkpoint into fp16 GGUF via llama.cpp's conversion script.
# This fp16 GGUF is the common starting point for both the PTQ and QAT export paths
# (Phase 2 / Phase 4 both quantize from a GGUF fp16 base to keep the pipeline consistent).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
CONVERT_SCRIPT="$LLAMA_DIR/convert_hf_to_gguf.py"
CONVERT_REQS="$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

HF_MODEL_DIR="${1:-models/qwen3-0.6b-hf}"
OUT_FILE="${2:-models/qwen3-0.6b-fp16.gguf}"

if [ ! -f "$CONVERT_SCRIPT" ]; then
  echo "ERROR: $CONVERT_SCRIPT not found. Run scripts/build_llama_cpp.sh first." >&2
  exit 1
fi

if [ -f "$CONVERT_REQS" ]; then
  echo "==> Installing llama.cpp's pinned conversion dependencies"
  pip install -r "$CONVERT_REQS"
fi

mkdir -p "$(dirname "$OUT_FILE")"

echo "==> Converting $HF_MODEL_DIR -> $OUT_FILE (f16)"
python3 "$CONVERT_SCRIPT" "$HF_MODEL_DIR" --outfile "$OUT_FILE" --outtype f16

echo "==> Done: $OUT_FILE"
ls -lh "$OUT_FILE"

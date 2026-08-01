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

# llama.cpp pins `torch==2.11.0` in requirements-convert_hf_to_gguf.txt behind
# `--extra-index-url https://download.pytorch.org/whl/cpu`. Installing that as-is
# REPLACES this project's hardware-specific torch (e.g. the ROCm build on the
# AMD GPU box) with a CPU-only wheel, which silently breaks every GPU path --
# train_qat.py, diagnose_qat_grads.py -- long after the conversion itself
# reported success. convert_hf_to_gguf.py only uses torch to *load* tensors,
# never for compute, so it has no stake in which build is installed.
#
# So: never let this file install torch. Everything else it pins is fine.
if [ -f "$CONVERT_REQS" ]; then
  if python3 -c "import numpy, gguf, sentencepiece, transformers" 2>/dev/null; then
    echo "==> Conversion dependencies already satisfied; skipping pip install"
  else
    echo "==> Installing llama.cpp's conversion dependencies (torch excluded)"
    # Written next to the original so its relative `-r ./requirements-*.txt`
    # include still resolves.
    FILTERED_REQS="$(dirname "$CONVERT_REQS")/.requirements-convert-no-torch.txt"
    trap 'rm -f "$FILTERED_REQS"' EXIT
    grep -vE '^[[:space:]]*(torch|--extra-index-url[[:space:]]+https://download\.pytorch\.org/whl/(cpu|nightly))' \
      "$CONVERT_REQS" > "$FILTERED_REQS" || true
    pip install -r "$FILTERED_REQS"
  fi
fi

if ! python3 -c "import torch" 2>/dev/null; then
  echo "ERROR: torch is not installed, and this script deliberately will not" >&2
  echo "install it (see comment above -- the pinned wheel is CPU-only)." >&2
  echo "Install the build matching this machine's accelerator first, e.g. ROCm:" >&2
  echo "  pip install torch --index-url \"\${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}\"" >&2
  echo "or see https://pytorch.org/get-started/locally/ for the current index URL." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"

echo "==> Converting $HF_MODEL_DIR -> $OUT_FILE (f16)"
python3 "$CONVERT_SCRIPT" "$HF_MODEL_DIR" --outfile "$OUT_FILE" --outtype f16

echo "==> Done: $OUT_FILE"
ls -lh "$OUT_FILE"

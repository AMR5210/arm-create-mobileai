#!/usr/bin/env bash
# Exports a QAT-trained HF checkpoint into GGUF Q2_K.
#
# This is now a thin wrapper around scripts/export_qat_gguf.py, which packs
# Q2_K in-process (a faithful port of ggml's reference quantiser) so we can:
#   Fix 1 - control the Q2_K scheme instead of black-box re-quantizing;
#   Fix 2 - keep the mixed-precision --skip-layers as F16 (not crush them to 2-bit);
#   Fix 3 - restore the base tokenizer before conversion.
# Unlike the old path it does NOT need a compiled llama-quantize; it only needs
# llama.cpp's pure-Python convert_hf_to_gguf.py (in third_party/llama.cpp) and
# the `gguf` pip package.
#
# Usage (preferred, explicit flags forwarded to the Python script):
#   scripts/export_qat_gguf.sh --hf-dir models/qwen3-0.6b-qat-hf \
#       --out models/qwen3-0.6b-qat-q2_k.gguf \
#       --skip-layers layers.16.self_attn.k_proj layers.27.mlp.gate_proj ...
#
# Legacy positional form (single HF dir) is still accepted for back-compat:
#   scripts/export_qat_gguf.sh models/qwen3-0.6b-qat-hf
# but note: pass --skip-layers to preserve mixed precision, otherwise every
# quantizable layer (including the ones trained full-precision) becomes Q2_K.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_EXPORT="$ROOT_DIR/scripts/export_qat_gguf.py"

if [ "$#" -ge 1 ] && [ "${1#-}" = "$1" ]; then
  # First arg is a bare path (legacy positional): map to --hf-dir, forward rest.
  HF_DIR="$1"; shift
  exec python3 "$PY_EXPORT" --hf-dir "$HF_DIR" "$@"
fi

exec python3 "$PY_EXPORT" "$@"

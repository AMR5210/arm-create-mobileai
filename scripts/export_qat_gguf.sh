#!/usr/bin/env bash
# Exports a QAT-trained HF checkpoint into the same GGUF Q2_K runtime format
# as the PTQ-2bit baseline, so speed/memory stay identical by construction
# and only quality (perplexity / instruction accuracy) can differ.
#
# Note: llama-quantize computes its own per-block scale/zero-point when
# producing Q2_K, so the exported weights are not bit-identical to what was
# simulated during QAT training (qat/fake_quant.py uses a simpler per-group
# affine scheme, not llama.cpp's K-quant super-block layout). The QAT premise
# still holds in practice -- gradient descent has shaped the weights to be
# more robust to 2-bit rounding in general, not just to the exact scheme used
# during training -- but this is a real limitation, not a rounding error, and
# should be stated plainly in the write-up rather than glossed over.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QAT_HF_DIR="${1:-models/qwen3-0.6b-qat-hf}"
FP16_OUT="${2:-models/qwen3-0.6b-qat-fp16.gguf}"
Q2K_OUT="${3:-models/qwen3-0.6b-qat-q2_k.gguf}"

"$ROOT_DIR/scripts/convert_to_gguf.sh" "$QAT_HF_DIR" "$FP16_OUT"
"$ROOT_DIR/scripts/quantize_ptq.sh" "$FP16_OUT" "$Q2K_OUT" Q2_K

echo
echo "==> QAT export complete: $Q2K_OUT"
echo "Benchmark it with:"
echo "  python scripts/benchmark.py --tag qat-2bit --device \"iPhone 17 Pro Max\" --model $Q2K_OUT"

#!/usr/bin/env bash
# Clones and builds llama.cpp. On Arm hosts (e.g. the Mac mini M4 dev machine),
# enables Arm's KleidiAI-accelerated matmul kernels.
#
# This builds a native binary for local development/iteration only. Cross-compiling
# for iOS (needed for on-device iPhone benchmarks) is a separate step handled later.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "==> Cloning llama.cpp"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  echo "==> Updating existing llama.cpp checkout"
  git -C "$LLAMA_DIR" pull --ff-only
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  echo "==> Arm host detected ($ARCH): enabling KleidiAI-accelerated kernels"
  CMAKE_ARGS+=(-DGGML_CPU_KLEIDIAI=ON)
else
  echo "==> Non-Arm host detected ($ARCH): building without KleidiAI (dev-only build, not for benchmark numbers)"
fi

echo "==> Configuring build"
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

echo "==> Building (this can take a few minutes)"
cmake --build "$BUILD_DIR" --config Release -j

echo
echo "Build complete. Key binaries in $BUILD_DIR/bin:"
echo "  llama-cli          - interactive/one-shot inference"
echo "  llama-bench        - throughput (tokens/sec) + memory benchmarking"
echo "  llama-perplexity   - perplexity evaluation over a text corpus"
echo "  llama-quantize     - PTQ conversion (e.g. to Q2_K)"
echo
echo "Verify fp16 inference works end-to-end, e.g.:"
echo "  $BUILD_DIR/bin/llama-cli -m <path-to-gguf> -p \"Hello\" -n 32"

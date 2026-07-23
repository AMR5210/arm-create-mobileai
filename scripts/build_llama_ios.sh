#!/usr/bin/env bash
# Cross-compiles llama.cpp into an .xcframework (iOS device + simulator, macOS)
# via llama.cpp's own build-xcframework.sh.
#
# Must run on macOS with Xcode + command line tools installed -- this is a
# Mac mini step. It cannot run in this repo's Linux dev sandbox, or on any
# non-macOS host.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
XCFRAMEWORK_SCRIPT="$LLAMA_DIR/build-xcframework.sh"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: this must run on macOS with Xcode installed -- run it on the Mac mini, not here." >&2
  exit 1
fi

if [ ! -d "$LLAMA_DIR" ]; then
  echo "ERROR: $LLAMA_DIR not found. Run scripts/build_llama_cpp.sh first (it clones llama.cpp)." >&2
  exit 1
fi

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "ERROR: xcodebuild not found. Install Xcode and its command line tools first." >&2
  exit 1
fi

if [ ! -f "$XCFRAMEWORK_SCRIPT" ]; then
  echo "ERROR: $XCFRAMEWORK_SCRIPT not found in this llama.cpp checkout." >&2
  echo "The script may have moved/been renamed upstream since this was written --" >&2
  echo "check the llama.cpp repo root (or its docs/build.md) for the current iOS/xcframework build entry point." >&2
  exit 1
fi

echo "==> Building llama.xcframework (iOS device + simulator, macOS) via llama.cpp's build-xcframework.sh"
( cd "$LLAMA_DIR" && ./build-xcframework.sh )

OUT_DIR="$LLAMA_DIR/build-apple/llama.xcframework"
echo
echo "==> Done: $OUT_DIR"
echo "Next: docs/IOS_BENCHMARK_HARNESS_SPEC.md specifies what the iOS app needs to"
echo "implement against this framework (build that part in a session with Xcode available)."

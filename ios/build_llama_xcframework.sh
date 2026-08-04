#!/usr/bin/env bash
# Cross-compiles llama.cpp into llama.xcframework for iOS (device + simulator)
# with Arm KleidiAI microkernels enabled.
#
# This wraps llama.cpp's build-xcframework.sh rather than calling it directly.
# That script hardcodes its cmake flags and cannot be steered by environment
# variables, and three of its defaults do not suit this project.
#
#   1. It does not pass -DGGML_CPU_KLEIDIAI, so KleidiAI is not compiled in.
#
#   2. It sets -DGGML_NATIVE=OFF, which is correct for cross-compiling since the
#      target CPU cannot be probed from the host. However,
#      ggml/src/ggml-cpu/CMakeLists.txt runs its -march feature detection only
#      when GGML_NATIVE is ON, so with it OFF and no GGML_CPU_ARM_ARCH set,
#      ARCH_FLAGS receives no -march. The KleidiAI block inspects ARCH_FLAGS for
#      "+dotprod"/"+i8mm"/"+sme" to select microkernels, so it then compiles the
#      pack helpers with no matmul kernels: ctx.kernels_* remain null and
#      supports_op() returns false for every tensor. Setting GGML_CPU_ARM_ARCH
#      explicitly is what makes the KleidiAI flag effective in practice.
#
#   3. It builds the simulator slice for "arm64;x86_64". An ARM -march is
#      invalid for the x86_64 compile ("unknown target CPU"), so the simulator
#      slice is restricted to arm64. Apple Silicon Macs run the arm64 simulator
#      natively, so coverage is unaffected.
#
# The macOS/visionOS/tvOS slices are also trimmed, reducing the build from seven
# slices to two.
#
# ARM_ARCH is a parameter rather than a constant: it is compiled into every ggml
# CPU source, so a binary built for a newer baseline raises SIGILL on an older
# chip. See the table below.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
UPSTREAM_SCRIPT="$LLAMA_DIR/build-xcframework.sh"
PATCHED_SCRIPT="$LLAMA_DIR/build-xcframework-kleidiai.sh"

# Which Arm baseline to compile every ggml CPU source against.
#
#   armv8.2-a+dotprod+fp16          A14 (iPhone 12) and newer. KleidiAI gets its
#                                   dotprod matmul kernels only.
#   armv8.6-a+dotprod+i8mm+fp16     A15 and newer. Adds KleidiAI's i8mm kernels.
#                                   SIGILLs on A14 -- do NOT ship this to an
#                                   iPhone 12.
#
# Default targets the project's primary benchmark device (A19 Pro). Build a
# second xcframework with the armv8.2-a value for any A14 comparison run.
ARM_ARCH="${ARM_ARCH:-armv8.6-a+dotprod+i8mm+fp16}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: must run on macOS with Xcode installed." >&2
  exit 1
fi
if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "ERROR: xcodebuild not found. Install Xcode + command line tools." >&2
  exit 1
fi
if [ ! -f "$UPSTREAM_SCRIPT" ]; then
  echo "ERROR: $UPSTREAM_SCRIPT not found." >&2
  echo "Run scripts/build_llama_cpp.sh first (it clones llama.cpp)." >&2
  exit 1
fi

echo "==> Xcode: $(xcodebuild -version | head -n1)"
echo "==> Arm baseline: -march=$ARM_ARCH"
echo

# ---------------------------------------------------------------------------
# Patch a copy of the upstream script. Every edit is asserted, so a change in
# upstream's structure surfaces here rather than producing a framework without
# KleidiAI.
# ---------------------------------------------------------------------------
assert_changed() {
  local label="$1" before="$2" after="$3"
  if [ "$before" = "$after" ]; then
    echo "ERROR: patch '$label' did not apply. Upstream build-xcframework.sh" >&2
    echo "       has changed structure; re-check it against this script." >&2
    exit 1
  fi
}

cp "$UPSTREAM_SCRIPT" "$PATCHED_SCRIPT"

# Patch 1: enable KleidiAI and pin the Arm baseline (see reason 1 + 2 above).
before="$(md5 -q "$PATCHED_SCRIPT")"
python3 - "$PATCHED_SCRIPT" "$ARM_ARCH" <<'PY'
import sys, re
path, arch = sys.argv[1], sys.argv[2]
src = open(path).read()
anchor = "    -DGGML_OPENMP=${GGML_OPENMP}\n"
if anchor not in src:
    sys.exit("anchor -DGGML_OPENMP not found in COMMON_CMAKE_ARGS")
src = src.replace(
    anchor,
    anchor
    + "    -DGGML_CPU_KLEIDIAI=ON\n"
    + f"    -DGGML_CPU_ARM_ARCH={arch}\n",
    1,
)
open(path, "w").write(src)
PY
assert_changed "enable KleidiAI + ARM arch" "$before" "$(md5 -q "$PATCHED_SCRIPT")"

# Patch 2: simulator slice arm64-only (see reason 3 above). The other
# multi-arch slices this touches are deleted by patch 3 anyway.
before="$(md5 -q "$PATCHED_SCRIPT")"
sed -i '' 's|-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"|-DCMAKE_OSX_ARCHITECTURES="arm64"|g' "$PATCHED_SCRIPT"
assert_changed "simulator arm64-only" "$before" "$(md5 -q "$PATCHED_SCRIPT")"

# Patch 3: drop the macOS / visionOS / tvOS slices.
before="$(md5 -q "$PATCHED_SCRIPT")"
python3 - "$PATCHED_SCRIPT" <<'PY'
import sys
path = sys.argv[1]
lines = open(path).read().splitlines(keepends=True)

start = next((i for i, l in enumerate(lines) if 'echo "Building for macOS..."' in l), None)
end   = next((i for i, l in enumerate(lines) if l.startswith("# Setup frameworks")), None)
if start is None or end is None or start >= end:
    sys.exit("could not locate the macOS..tvOS build block to delete")
del lines[start:end]

# Removes the setup_framework_structure / combine_static_libraries calls and
# the -framework/-debug-symbols argument pairs for the deleted platforms.
lines = [l for l in lines
         if not any(k in l for k in ("build-macos", "build-visionos", "build-tvos"))]
open(path, "w").write("".join(lines))
PY
assert_changed "drop non-iOS slices" "$before" "$(md5 -q "$PATCHED_SCRIPT")"

# Confirm the flags are present in the script about to be run.
for needle in "-DGGML_CPU_KLEIDIAI=ON" "-DGGML_CPU_ARM_ARCH=$ARM_ARCH"; do
  grep -qF -- "$needle" "$PATCHED_SCRIPT" || { echo "ERROR: '$needle' missing from patched script" >&2; exit 1; }
done
for stale in "build-macos" "build-visionos" "build-tvos"; do
  ! grep -qF -- "$stale" "$PATCHED_SCRIPT" || { echo "ERROR: '$stale' still referenced" >&2; exit 1; }
done
echo "==> Patched build script verified: $PATCHED_SCRIPT"
echo

chmod +x "$PATCHED_SCRIPT"
( cd "$LLAMA_DIR" && ./build-xcframework-kleidiai.sh )

OUT="$LLAMA_DIR/build-apple/llama.xcframework"
if [ ! -d "$OUT" ]; then
  echo "ERROR: expected $OUT but it was not produced." >&2
  exit 1
fi

echo
echo "==> Built $OUT"

# The simulator slice is compiled arm64-only, while upstream still emits an empty
# x86_64 dylib slice. Dropping it keeps the framework from advertising an
# architecture it cannot link.
"$ROOT_DIR/ios/strip_dead_sim_arch.sh" "$OUT"

echo
echo "==> Verifying KleidiAI is actually linked in (not just compiled out):"
"$ROOT_DIR/ios/verify_kleidiai.sh" "$OUT"

# Stage the framework next to the app project so the Xcode project does not
# reach into third_party/ (which is gitignored and rebuilt from scratch).
STAGED="$ROOT_DIR/ios/Frameworks"
mkdir -p "$STAGED"
rm -rf "$STAGED/llama.xcframework"
cp -R "$OUT" "$STAGED/llama.xcframework"
echo
echo "==> Staged framework at ios/Frameworks/llama.xcframework"

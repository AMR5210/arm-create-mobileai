#!/usr/bin/env bash
# Confirms a built llama.xcframework contains KleidiAI matmul microkernels rather
# than KleidiAI glue with an empty kernel table, which is the result when -march
# carries no +dotprod/+i8mm/+sme (see build_llama_xcframework.sh).
#
# A successful build with KleidiAI enabled occurs in both cases, so the presence
# of kai_* matmul symbols in the binary is the check that distinguishes them.
#
# Usage: verify_kleidiai.sh [path/to/llama.xcframework]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XC="${1:-$ROOT_DIR/third_party/llama.cpp/build-apple/llama.xcframework}"

if [ ! -d "$XC" ]; then
  echo "ERROR: $XC not found. Run ios/build_llama_xcframework.sh first." >&2
  exit 1
fi

status=0
found_any=0

for slice_dir in "$XC"/*/; do
  slice="$(basename "$slice_dir")"
  bin="$slice_dir/llama.framework/llama"
  [ -f "$bin" ] || continue
  found_any=1

  echo "--- $slice"
  echo "    archs:      $(lipo -archs "$bin" 2>/dev/null || echo '?')"

  glue=$(nm -gU "$bin" 2>/dev/null | grep -c "kleidiai" || true)
  dot=$(nm -U "$bin" 2>/dev/null  | grep -c "kai_run_matmul.*dotprod" || true)
  i8mm=$(nm -U "$bin" 2>/dev/null | grep -c "kai_run_matmul.*i8mm" || true)
  sme=$(nm -U "$bin" 2>/dev/null  | grep -c "kai_run_matmul.*sme" || true)
  total=$(nm -U "$bin" 2>/dev/null | grep -c "kai_run_matmul" || true)

  echo "    kleidiai glue symbols:   $glue"
  echo "    kai_run_matmul total:    $total   (dotprod=$dot i8mm=$i8mm sme=$sme)"

  if [ "$total" -eq 0 ]; then
    echo "    RESULT: FAIL - no KleidiAI matmul microkernels in this slice."
    echo "            KleidiAI would not be dispatched at runtime."
    status=1
  else
    echo "    RESULT: OK - KleidiAI microkernels present."
  fi
  echo
done

if [ "$found_any" -eq 0 ]; then
  echo "ERROR: no framework binaries found inside $XC" >&2
  exit 1
fi

exit $status

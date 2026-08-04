#!/usr/bin/env bash
# Removes the empty x86_64 stub from the simulator slice of llama.xcframework.
#
# build_llama_xcframework.sh compiles the simulator slice for arm64 only, since an
# ARM -march is invalid for an x86_64 compile. Upstream's combine_static_libraries
# emits an x86_64 dylib slice regardless, leaving the framework advertising an
# architecture that contains no llama/kai symbols; linking an app against it would
# fail with undefined symbols.
#
# Stripping the slice and correcting Info.plist keeps the published artifact's
# declared architectures accurate. Idempotent: re-running on an already-stripped
# framework is a no-op.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XC="${1:-$ROOT_DIR/third_party/llama.cpp/build-apple/llama.xcframework}"
PB=/usr/libexec/PlistBuddy

if [ ! -d "$XC" ]; then
  echo "ERROR: $XC not found." >&2
  exit 1
fi

OLD_ID="ios-arm64_x86_64-simulator"
NEW_ID="ios-arm64-simulator"

if [ ! -d "$XC/$OLD_ID" ]; then
  if [ -d "$XC/$NEW_ID" ]; then
    echo "==> Simulator slice already arm64-only, nothing to strip."
    exit 0
  fi
  echo "ERROR: neither $OLD_ID nor $NEW_ID found in $XC" >&2
  exit 1
fi

BIN="$XC/$OLD_ID/llama.framework/llama"
archs="$(lipo -archs "$BIN")"

if [[ "$archs" == *x86_64* ]]; then
  # Strip only when the slice is genuinely empty, so a future build that does
  # produce working x86_64 code is preserved.
  sym_count=$(nm -arch x86_64 -U "$BIN" 2>/dev/null | grep -c "llama_decode" || true)
  if [ "$sym_count" -ne 0 ]; then
    echo "ERROR: x86_64 slice contains $sym_count symbol(s); leaving it in place." >&2
    echo "       The simulator slice is now being built for x86_64;" >&2
    echo "       revisit the arm64-only patch in build_llama_xcframework.sh." >&2
    exit 1
  fi
  echo "==> Stripping empty x86_64 slice from simulator binary"
  lipo -remove x86_64 "$BIN" -output "$BIN.tmp"
  mv "$BIN.tmp" "$BIN"
fi

echo "==> Renaming slice directory: $OLD_ID -> $NEW_ID"
mv "$XC/$OLD_ID" "$XC/$NEW_ID"

echo "==> Correcting Info.plist"
# Locate the simulator entry by index; order is not guaranteed, so search.
count=$($PB -c "Print :AvailableLibraries" "$XC/Info.plist" | grep -c "Dict {" || true)
for ((i = 0; i < count; i++)); do
  id=$($PB -c "Print :AvailableLibraries:$i:LibraryIdentifier" "$XC/Info.plist" 2>/dev/null || echo "")
  if [ "$id" = "$OLD_ID" ]; then
    $PB -c "Set :AvailableLibraries:$i:LibraryIdentifier $NEW_ID" "$XC/Info.plist"
    # Rebuild SupportedArchitectures as [arm64].
    $PB -c "Delete :AvailableLibraries:$i:SupportedArchitectures" "$XC/Info.plist"
    $PB -c "Add :AvailableLibraries:$i:SupportedArchitectures array" "$XC/Info.plist"
    $PB -c "Add :AvailableLibraries:$i:SupportedArchitectures:0 string arm64" "$XC/Info.plist"
    echo "    updated entry $i"
    break
  fi
done

echo "==> Done. Slices now:"
for d in "$XC"/*/; do
  b="$d/llama.framework/llama"
  [ -f "$b" ] && echo "    $(basename "$d"): $(lipo -archs "$b")"
done

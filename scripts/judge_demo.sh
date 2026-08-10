#!/usr/bin/env bash
# End-to-end validation of the 2-bit QAT model in one command.
#
# THIS USES THE DESKTOP llama.cpp BUILD, NOT THE iOS HARNESS.
#
# Runs on any Arm64 or x86_64 host with Bash, Git, CMake and a C++ toolchain --
# Apple Silicon, Intel Mac, or Linux. It does NOT require an iPhone, Xcode, a
# provisioning profile, or code signing. Nothing here touches ios/.
#
# The on-device figures in README.md (iPhone 17 Pro Max: 67.30 tok/s generation,
# 2181.0 MB peak RAM) come from a separate harness under ios/, which does need a
# Mac with Xcode, a signing identity and a physical device. This script exists so
# the model itself can be verified and exercised without any of that. The
# throughput it prints is this host's, not the recorded iPhone number, and the two
# are not comparable.
#
# What it does, skipping any step already satisfied:
#   1. Builds the pinned llama.cpp revision (b10210) if the binary is absent.
#   2. Downloads the QAT GGUF if absent -- that file only, not all three variants.
#   3. Verifies SHA-256 against the published digest, and hard-fails on mismatch.
#   4. Runs one fixed prompt and prints the answer, throughput, and the digest.
#
# Safe to re-run. A warm run prints its result in a few seconds; the first run on
# a clean checkout spends several minutes compiling llama.cpp.
#
# Usage:  ./scripts/judge_demo.sh
#         ./scripts/judge_demo.sh "Your prompt"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$ROOT_DIR/third_party/llama.cpp/build/bin/llama-cli"
MODEL="$ROOT_DIR/models/qwen3-0.6b-qat-q2_k.gguf"
LOG="$ROOT_DIR/results/judge_demo.log"

# Published digest of the featured fineweb-blend checkpoint. A filename is not an
# identity: a superseded export once occupied this exact path and invalidated a
# round of measurements. See scripts/verify_model_signatures.py.
EXPECTED_SHA="a861b8924a2b1881720d38123ec32aae7f99ac05732d819644a6584f3cc38fef"
MODEL_URL="https://huggingface.co/AMR5210/qwen3-0.6b-qat-q2k/resolve/main/fineweb-blend/qwen3-0.6b-qat-fineweb-blend-q2_k.gguf"

PROMPT="${1:-What is the capital of France? Answer in one sentence.}"
N_PREDICT=64

RULE="------------------------------------------------------------------"

die() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

# shasum on macOS, sha256sum on most Linux distributions.
sha256_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else die "neither shasum nor sha256sum found; cannot verify the model"
  fi
}

step() { printf '  [%s/3] %-28s %s\n' "$1" "$2" "$3"; }

mkdir -p "$ROOT_DIR/results" "$ROOT_DIR/models"
: >"$LOG"

printf '\n==================================================================\n'
printf '  2-bit QAT validation -- Qwen3-0.6B, desktop llama.cpp\n'
printf '==================================================================\n'

# ---------------------------------------------------------------- 1. toolchain
if [ -x "$CLI" ]; then
  step 1 "llama.cpp (b10210)" "reusing existing build"
else
  step 1 "llama.cpp (b10210)" "building -- several minutes, output -> ${LOG#$ROOT_DIR/}"
  if ! "$ROOT_DIR/scripts/build_llama_cpp.sh" >>"$LOG" 2>&1; then
    tail -20 "$LOG" >&2
    die "llama.cpp build failed. Full log: $LOG"
  fi
  [ -x "$CLI" ] || die "build reported success but $CLI is missing. Log: $LOG"
  step 1 "llama.cpp (b10210)" "built"
fi

# ------------------------------------------------------------------- 2. model
PREEXISTING=1
if [ -f "$MODEL" ]; then
  step 2 "QAT model" "reusing $(basename "$MODEL")"
else
  PREEXISTING=0
  command -v curl >/dev/null 2>&1 || die "curl not found; cannot download the model"
  step 2 "QAT model" "downloading 472 MiB, no account or token needed"
  # Downloaded to a temporary path and only moved into place after the digest
  # check, so models/ never holds an unverified file. The destination filename
  # differs from the published one by design: every script here and the iOS app
  # read one fixed path regardless of which variant is current.
  TMP="$MODEL.partial.$$"
  trap 'rm -f "$TMP"' EXIT
  if ! curl -fL --retry 2 -o "$TMP" "$MODEL_URL" >>"$LOG" 2>&1; then
    die "download failed. Full log: $LOG"
  fi
fi

# ----------------------------------------------------------------- 3. identity
TARGET="${TMP:-$MODEL}"
GOT="$(sha256_of "$TARGET")"
if [ "$GOT" != "$EXPECTED_SHA" ]; then
  printf '\n  [3/3] %-28s MISMATCH\n' "SHA-256"
  printf '\nERROR: model identity check failed. Refusing to continue.\n' >&2
  printf '  expected: %s\n' "$EXPECTED_SHA" >&2
  printf '  actual  : %s\n' "$GOT" >&2
  printf '  file    : %s (%s bytes)\n' "$TARGET" "$(wc -c <"$TARGET" | tr -d ' ')" >&2
  if [ "$(wc -c <"$TARGET" | tr -d ' ')" -lt 1000000 ]; then
    printf '\n  The file is far too small to be a 472 MiB GGUF. A failed download\n' >&2
    printf '  writes an HTML or JSON error body under the same .gguf name.\n' >&2
  fi
  if [ "$PREEXISTING" = "1" ]; then
    printf '\n  This file was already on disk, so it was left untouched. It is not the\n' >&2
    printf '  published checkpoint. Move it aside and re-run to fetch a clean copy.\n' >&2
  fi
  exit 1
fi

if [ "$PREEXISTING" = "0" ]; then
  mv "$TMP" "$MODEL"
  trap - EXIT
fi
step 3 "SHA-256" "verified"

SIZE_MIB="$(awk "BEGIN{printf \"%.1f\", $(wc -c <"$MODEL" | tr -d ' ')/1048576}")"
printf '\n  SHA-256 verified: %s\n' "$EXPECTED_SHA"
printf '  Model: %s\n' "${MODEL#$ROOT_DIR/}"
printf '  Format: GGUF Q2_K, 2-bit weights, %s MiB on disk\n' "$SIZE_MIB"

# ---------------------------------------------------------------- 4. inference
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
# --log-disable does not suppress llama-cli's banner or prompt echo, so the answer
# is taken from between the echoed prompt line and the throughput footer rather
# than by scanning the whole stream.
"$CLI" -m "$MODEL" \
  --jinja --chat-template-kwargs '{"enable_thinking":false}' \
  -p "$PROMPT" -n "$N_PREDICT" --temp 0 --top-k 1 \
  --no-warmup -st --log-disable </dev/null >"$OUT" 2>>"$LOG" || true

ANSWER="$(awk '/^> /{f=1;next} /^\[ Prompt:/{f=0} f' "$OUT" | sed '/^[[:space:]]*$/d')"
PERF="$(grep -m1 '^\[ Prompt:' "$OUT" || true)"
PP="$(printf '%s' "$PERF" | sed -n 's/.*Prompt: *\([0-9.]*\).*/\1/p')"
TG="$(printf '%s' "$PERF" | sed -n 's/.*Generation: *\([0-9.]*\).*/\1/p')"

[ -n "$ANSWER" ] || die "no output produced by llama-cli. Log: $LOG"

printf '\n%s\n' "$RULE"
printf '  Prompt\n    %s\n' "$PROMPT"
printf '\n  Answer from the 2-bit QAT model\n'
printf '%s\n' "$ANSWER" | sed 's/^/    /'
if [ -n "$PP" ] && [ -n "$TG" ]; then
  printf '\n  Throughput on this host\n    prompt %s tok/s    generation %s tok/s\n' "$PP" "$TG"
fi
printf '%s\n' "$RULE"
printf '  Host: %s %s -- desktop CPU build. No iPhone, Xcode or signing used.\n' "$(uname -s)" "$(uname -m)"
printf '  The recorded iPhone 17 Pro Max figures come from ios/; see README.md.\n\n'

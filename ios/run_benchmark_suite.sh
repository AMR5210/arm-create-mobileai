#!/usr/bin/env bash
# Drives the LlamaBench five-metric suite across model variants on a simulator,
# one variant per app process.
#
# Launch and process handling follow three requirements of the simctl interface:
#
#   1. `simctl launch --console-pty` returns when the app exits, not when its work
#      completes, and a SwiftUI app does not exit on its own. Launches are
#      therefore backgrounded, and completion is detected by polling the app's log
#      for its end-of-run marker.
#
#   2. A `pgrep -f` pattern naming the app also matches the calling shell, whose
#      command line contains that pattern. Process checks use a bracketed pattern,
#      which matches the app alone.
#
#   3. Peak RAM is only meaningful per variant when each runs in a fresh process,
#      since a process that has already loaded a model carries its allocations.
#      The app is terminated and confirmed exited between variants.
#
# Each variant writes results/<tag>.json under --out-dir as it completes, so an
# interruption costs at most the variant in flight. Perplexity additionally
# checkpoints within a variant (see BenchmarkSuite.saveCheckpoint).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE_NAME="${DEVICE_NAME:-iPhone 17 Pro Max}"
APP_ID="org.armcreate.llamabench"
APP_PATH="${APP_PATH:-$ROOT_DIR/ios/LlamaBench/build/DerivedData/Build/Products/Debug-iphonesimulator/LlamaBench.app}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/dev-ios-sim}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/eval/data}"
VARIANTS="${VARIANTS:-baseline-fp16 ptq-2bit qat-2bit}"
BACKEND="${BACKEND:-cpu}"
# Generous ceiling: a full 583-chunk perplexity pass plus 100-question
# instruction eval runs about an hour per variant on the simulator.
PER_VARIANT_TIMEOUT="${PER_VARIANT_TIMEOUT:-9000}"
LOG="${LOG:-$OUT_DIR/run.log}"

mkdir -p "$OUT_DIR"

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

app_running() {
    # Bracketed pattern, per requirement 2: matches the app and not this shell.
    pgrep -f "CoreSimulator/Devices/.*LlamaBench[.]app" >/dev/null 2>&1
}

stop_app() {
    xcrun simctl terminate "$UDID" "$APP_ID" >/dev/null 2>&1 || true
    local waited=0
    while app_running; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 30 ]; then
            say "  terminate did not take after ${waited}s; force-killing"
            pkill -f "CoreSimulator/Devices/.*LlamaBench[.]app" || true
            sleep 2
            break
        fi
    done
    if app_running; then
        say "  ERROR: app still running; cannot start a clean variant"
        return 1
    fi
    return 0
}

# --- preflight ---------------------------------------------------------------
if [ ! -d "$APP_PATH" ]; then
    say "ERROR: app bundle not found at $APP_PATH -- build it first."
    exit 1
fi

UDID="$(xcrun simctl list devices available \
        | awk -v n="$DEVICE_NAME" -F'[()]' '$0 ~ n"[ ]*\\(" {print $2; exit}')"
if [ -z "${UDID:-}" ]; then
    say "ERROR: no available simulator named '$DEVICE_NAME'"
    exit 1
fi

say "=== LlamaBench suite ==="
say "device:   $DEVICE_NAME ($UDID)"
say "app:      $APP_PATH"
say "out:      $OUT_DIR"
say "variants: $VARIANTS"

xcrun simctl bootstatus "$UDID" -b >/dev/null 2>&1 || true
stop_app || exit 1
xcrun simctl install "$UDID" "$APP_PATH" || { say "ERROR: install failed"; exit 1; }
say "installed"

# --- per-variant runs --------------------------------------------------------
failed=""
for v in $VARIANTS; do
    vlog="$OUT_DIR/$v.run.log"
    rm -f "$vlog"
    say "--- $v : starting (log: $(basename "$vlog"))"

    stop_app || { failed="$failed $v"; continue; }

    # Backgrounded, per requirement 1 at the top of this file.
    env \
      SIMCTL_CHILD_LLAMABENCH_AUTORUN=1 \
      SIMCTL_CHILD_LLAMABENCH_MODE=bench \
      SIMCTL_CHILD_LLAMABENCH_VARIANT="$v" \
      SIMCTL_CHILD_LLAMABENCH_BACKEND="$BACKEND" \
      SIMCTL_CHILD_LLAMABENCH_MODEL_DIR="$MODEL_DIR" \
      SIMCTL_CHILD_LLAMABENCH_EVAL_DIR="$EVAL_DIR" \
      SIMCTL_CHILD_LLAMABENCH_OUT_DIR="$OUT_DIR" \
      xcrun simctl launch --console-pty "$UDID" "$APP_ID" >"$vlog" 2>&1 &
    launch_pid=$!

    waited=0
    while :; do
        if grep -qE "=== $v (COMPLETE|FAILED) ===" "$vlog" 2>/dev/null; then
            break
        fi
        if [ "$waited" -ge "$PER_VARIANT_TIMEOUT" ]; then
            say "  TIMEOUT after ${waited}s"
            break
        fi
        if ! app_running && [ "$waited" -gt 30 ]; then
            say "  app exited without a completion marker"
            break
        fi
        sleep 10
        waited=$((waited + 10))
    done

    kill "$launch_pid" 2>/dev/null || true
    stop_app || true

    if grep -q "=== $v COMPLETE ===" "$vlog" 2>/dev/null; then
        ppl=$(grep -oE "perplexity [0-9.]+" "$vlog" | tail -1)
        pr=$(grep -oE "forced-choice accuracy [0-9.]+%" "$vlog" | tail -1)
        say "  $v OK after ${waited}s  ($ppl, $pr)"
    else
        say "  $v DID NOT COMPLETE"
        failed="$failed $v"
    fi
done

say "=== suite finished ==="
if [ -n "$failed" ]; then
    say "incomplete variants:$failed"
    exit 1
fi
say "all variants completed"

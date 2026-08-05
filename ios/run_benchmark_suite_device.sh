#!/usr/bin/env bash
# Drives the LlamaBench five-metric suite across model variants on a physical
# device, one variant per app process.
#
# Device counterpart to run_benchmark_suite.sh. The differences from the simulator
# path come from devicectl:
#
#   1. `devicectl device process launch --console` streams output for the app's
#      lifetime and does not return when the app's work finishes, so launches are
#      backgrounded and completion is detected by polling the app's log for its
#      end-of-run marker.
#
#   2. `--terminate-existing` ends any prior instance as part of the launch,
#      giving each variant a fresh process. This matters for peak RAM, which is
#      only meaningful per variant when the process has not already loaded a model.
#
#   3. Models, eval corpora and results all live inside the app's data container.
#      Inputs are staged with `devicectl device copy to` before this script runs;
#      records are pulled back afterwards with `devicectl device copy from`.
#
# The app writes results/<tag>.json inside its Documents directory as each variant
# completes, and perplexity checkpoints every 25 chunks, so an interruption costs
# at most the variant in flight.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE_ID="${DEVICE_ID:?set DEVICE_ID to the devicectl device identifier}"
APP_ID="org.armcreate.llamabench"
VARIANTS="${VARIANTS:-baseline-fp16 ptq-2bit qat-2bit}"
BACKEND="${BACKEND:-cpu}"
# A full 583-chunk perplexity pass plus the 100-question instruction eval runs
# well under an hour per variant on A19-class hardware; this is a generous ceiling.
PER_VARIANT_TIMEOUT="${PER_VARIANT_TIMEOUT:-7200}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/device-iphone17promax}"
LOG="${LOG:-$OUT_DIR/run.log}"

mkdir -p "$OUT_DIR"
say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

say "=== LlamaBench device suite ==="
say "device:   $DEVICE_ID"
say "variants: $VARIANTS"
say "out:      $OUT_DIR"

failed=""
for v in $VARIANTS; do
    vlog="$OUT_DIR/$v.run.log"
    rm -f "$vlog"
    say "--- $v : starting"

    # Backgrounded, per note 1 above. --terminate-existing supplies the fresh
    # process required by note 2.
    xcrun devicectl device process launch \
        --device "$DEVICE_ID" --console --terminate-existing \
        --environment-variables "{\"LLAMABENCH_AUTORUN\":\"1\",\
\"LLAMABENCH_MODE\":\"bench\",\
\"LLAMABENCH_VARIANT\":\"$v\",\
\"LLAMABENCH_BACKEND\":\"$BACKEND\"}" \
        "$APP_ID" >"$vlog" 2>&1 &
    launch_pid=$!

    waited=0
    while :; do
        if grep -qE "=== $v (COMPLETE|FAILED) ===" "$vlog" 2>/dev/null; then break; fi
        if grep -qiE "invalid code signature|not been explicitly trusted" "$vlog" 2>/dev/null; then
            say "  launch denied: the developer profile is not trusted on the device"
            break
        fi
        if [ "$waited" -ge "$PER_VARIANT_TIMEOUT" ]; then
            say "  TIMEOUT after ${waited}s"
            break
        fi
        sleep 10
        waited=$((waited + 10))
    done

    kill "$launch_pid" 2>/dev/null || true

    # Memory pressure is worth surfacing per variant: the fp16 model alone is
    # 1.4 GB, and a jetsam kill would end the pass without a completion marker.
    if grep -qiE "jetsam|terminated due to memory|memory pressure|was killed" "$vlog" 2>/dev/null; then
        say "  MEMORY WARNING present in this variant's log -- inspect $vlog"
    fi

    if grep -q "=== $v COMPLETE ===" "$vlog" 2>/dev/null; then
        ppl=$(grep -oE "perplexity [0-9.]+" "$vlog" | tail -1)
        acc=$(grep -oE "forced-choice accuracy [0-9.]+%" "$vlog" | tail -1)
        ram=$(grep -oE "peak RAM: [0-9.]+ MB" "$vlog" | tail -1)
        say "  $v OK after ${waited}s  ($ppl, $acc, $ram)"
    else
        say "  $v DID NOT COMPLETE"
        failed="$failed $v"
    fi
done

# Pull the records off the device into OUT_DIR.
say "retrieving records from the device"
for v in $VARIANTS; do
    xcrun devicectl device copy from --device "$DEVICE_ID" \
        --domain-type appDataContainer --domain-identifier "$APP_ID" \
        --source "Documents/results/$v.json" --destination "$OUT_DIR/$v.json" \
        >/dev/null 2>&1 && say "  retrieved $v.json" || say "  could not retrieve $v.json"
done

say "=== suite finished ==="
if [ -n "$failed" ]; then
    say "incomplete variants:$failed"
    exit 1
fi
say "all variants completed"

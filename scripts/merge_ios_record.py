#!/usr/bin/env python3
"""Fold selected metrics from a partial on-device record into a complete one.

The iOS suite measures five metrics per variant and writes one record. Some
metrics are far more expensive than others: a full perplexity pass is tens of
minutes, while instruction eval is a couple of minutes. When only the cheap
metric needs re-running, the suite writes a partial record to its own filename
and this merges the relevant fields in, so the expensive measurement is not
repeated.

Guards against merging unrelated measurements: the two records must agree on
`tag` and `model_sha256`, since a metric measured against a different checkpoint
does not belong in the same record.

Example:
    scripts/merge_ios_record.py \\
        --into results/dev-ios-sim/baseline-fp16.json \\
        --from results/dev-ios-sim/baseline-fp16.instr.json \\
        --fields instruction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELD_GROUPS = {
    "instruction": [
        "instruction_forced_choice_accuracy",
        "instruction_per_subject_forced_choice_accuracy",
    ],
    "throughput": [
        "prompt_tokens_per_sec",
        "gen_tokens_per_sec",
    ],
    "perplexity": [
        "perplexity",
    ],
}
# Harness sub-keys carried across with their metric, so the record's methodology
# notes stay consistent with the values they describe.
HARNESS_KEYS = {
    "instruction": [
        "instruction_eval_prompt_format",
        "instruction_eval_total",
        "instruction_eval_method",
        "instruction_eval_forced_choice_correct",
        "instruction_eval_forced_choice_predictions",
        "instruction_eval_forced_choice_histogram",
        "instruction_eval_parse_rate_diagnostic",
        "instruction_eval_parsed",
        "instruction_eval_accuracy_diagnostic",
        "instruction_eval_per_subject_accuracy_diagnostic",
        "instruction_eval_metric_note",
    ],
    "throughput": [
        "throughput_method",
        "throughput_prompt_samples",
        "throughput_gen_samples",
    ],
    "perplexity": [
        "perplexity_method",
        "perplexity_corpus",
        "perplexity_chunks",
        "perplexity_tokens_scored",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", type=Path, required=True, help="record to update")
    ap.add_argument("--from", dest="src", type=Path, required=True, help="partial record")
    ap.add_argument("--fields", nargs="+", required=True, choices=sorted(FIELD_GROUPS),
                    help="metric groups to copy")
    ap.add_argument("--note", help="text appended to the target's merge_history")
    ap.add_argument("--allow-sha-mismatch", action="store_true")
    args = ap.parse_args()

    dst = json.loads(args.into.read_text())
    src = json.loads(args.src.read_text())

    if dst.get("tag") != src.get("tag"):
        return fail(f"tag mismatch: {dst.get('tag')!r} vs {src.get('tag')!r}")
    if not args.allow_sha_mismatch and dst.get("model_sha256") != src.get("model_sha256"):
        return fail(
            f"model_sha256 mismatch:\n  into: {dst.get('model_sha256')}\n  from: {src.get('model_sha256')}\n"
            "The two measurements are of different files; pass --allow-sha-mismatch only if that is intended."
        )

    obsolete = {"instruction": ["instruction_accuracy", "instruction_per_subject_accuracy",
                                "instruction_parse_rate", "instruction_per_subject_parse_rate"]}

    changed = []
    for group in args.fields:
        for key in obsolete.get(group, []):
            if dst.pop(key, None) is not None:
                changed.append(f"{key}: removed (superseded)")
        for key in FIELD_GROUPS[group]:
            if key in src:
                if dst.get(key) != src[key]:
                    changed.append(f"{key}: {dst.get(key)!r} -> {src[key]!r}")
                dst[key] = src[key]
        for hk in HARNESS_KEYS[group]:
            if hk in src.get("harness", {}):
                dst.setdefault("harness", {})[hk] = src["harness"][hk]

    history = dst.setdefault("merge_history", [])
    history.append({
        "merged_fields": args.fields,
        "source": str(args.src),
        "source_timestamp_utc": src.get("timestamp_utc"),
        "note": args.note,
    })

    args.into.write_text(json.dumps(dst, indent=2, sort_keys=True) + "\n")
    print(f"merged {args.fields} from {args.src} into {args.into}")
    for c in changed:
        print(f"  {c}")
    return 0


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

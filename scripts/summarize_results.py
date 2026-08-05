#!/usr/bin/env python3
"""Combines every results/*.json record into one comparison table
(results/summary.md and results/summary.csv).
"""
import csv
import json
from pathlib import Path

RESULTS_DIR = Path("results")

COLUMNS = [
    ("tag", "Tag"),
    ("device", "Device"),
    ("disk_mb", "Disk (MB)"),
    ("ram_mb", "Peak RAM (MB)"),
    ("prompt_tokens_per_sec", "Prompt tok/s"),
    ("gen_tokens_per_sec", "Gen tok/s"),
    ("perplexity", "Perplexity"),
    ("instruction_forced_choice_accuracy", "Instr. acc. (forced)"),
]


def load_records() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        # results/ also holds analysis files that are not per-variant benchmark
        # records (wikitext2_perplexity.json, c4_perplexity.json). They have no
        # "tag" key, so skip them rather than raising KeyError on the glob.
        if "tag" not in data:
            continue
        rows.append(
            {
                "tag": data["tag"],
                "device": data["device"],
                "disk_mb": round(data["disk_bytes"] / 1e6, 1),
                "ram_mb": round(data["peak_ram_bytes"] / 1e6, 1),
                "prompt_tokens_per_sec": _fmt(data["prompt_tokens_per_sec"]),
                "gen_tokens_per_sec": _fmt(data["gen_tokens_per_sec"]),
                "perplexity": _fmt(data["perplexity"]),
                "instruction_forced_choice_accuracy": (
                    f"{data['instruction_forced_choice_accuracy']:.1%}"
                    if data.get("instruction_forced_choice_accuracy") is not None
                    else "n/a"
                ),
            }
        )
    return rows


def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def write_csv(rows: list[dict]) -> None:
    out_path = RESULTS_DIR / "summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def write_markdown(rows: list[dict]) -> None:
    out_path = RESULTS_DIR / "summary.md"
    headers = [label for _, label in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key, _ in COLUMNS) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    rows = load_records()
    if not rows:
        print(f"No results found in {RESULTS_DIR}/*.json")
        return
    write_csv(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()

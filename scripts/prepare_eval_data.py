#!/usr/bin/env python3
"""Prepare evaluation data: a WikiText-2 perplexity corpus and a small
instruction-following eval slice drawn from MMLU.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

DEFAULT_OUT_DIR = Path("eval") / "data"
DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "high_school_computer_science",
    "elementary_mathematics",
    "logical_fallacies",
]


def prepare_wikitext2(out_dir: Path) -> None:
    print("==> Downloading wikitext-2-raw-v1 (test split)")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    out_path = out_dir / "wikitext2_test.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"    wrote {out_path} ({len(text):,} characters)")


def prepare_instruction_eval(out_dir: Path, n_per_subject: int, subjects: list[str]) -> None:
    print(f"==> Building instruction-eval slice from MMLU subjects: {', '.join(subjects)}")
    rows = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i, row in enumerate(ds):
            if i >= n_per_subject:
                break
            rows.append(
                {
                    "subject": subject,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],  # index into choices
                }
            )
    out_path = out_dir / "instruction_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"    wrote {out_path} ({len(rows)} questions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-subject", type=int, default=25)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepare_wikitext2(args.out_dir)
    prepare_instruction_eval(args.out_dir, args.n_per_subject, args.subjects)


if __name__ == "__main__":
    main()

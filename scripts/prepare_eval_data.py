#!/usr/bin/env python3
"""Prepare evaluation data: a WikiText-2 perplexity corpus, an optional C4
perplexity corpus, and a small instruction-following eval slice from MMLU.
"""
import argparse
import gzip
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_OUT_DIR = Path("eval") / "data"
# Matches wikitext2_test.txt's size (~1.29 M chars => ~583 chunks at n_ctx 512),
# so C4 and WikiText-2 perplexities are computed over a comparable number of
# chunks and their error bars are of similar width.
DEFAULT_C4_TARGET_CHARS = 1_290_000
DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "high_school_computer_science",
    "elementary_mathematics",
    "logical_fallacies",
]


def prepare_wikitext2(out_dir: Path) -> None:
    print("==> Downloading wikitext-2-raw-v1 (test split)")
    # Use the namespaced repo id: newer huggingface_hub (1.x) rejects the
    # legacy canonical id "wikitext" (must be "namespace/name"). The namespaced
    # id also resolves on older datasets versions, so this is safe everywhere.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    out_path = out_dir / "wikitext2_test.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"    wrote {out_path} ({len(text):,} characters)")


def prepare_c4(out_dir: Path, target_chars: int = DEFAULT_C4_TARGET_CHARS) -> None:
    """A C4 (en) validation slice, as a perplexity corpus that does NOT overlap
    WikiText.

    Why it exists: QAT here trains on a WikiText-2 *train* blend and is scored on
    WikiText-2 *test*. There is no leakage (different splits), but the training
    domain still matches the eval domain, so a WikiText-2 win partly reflects
    domain alignment. C4 is web text from a different distribution and nothing in
    the training mix comes from it, which makes it the check on whether a
    WikiText-2 gain actually generalizes.

    Downloads ONE validation shard rather than streaming the whole dataset (C4 en
    is ~300 GB), and takes documents in file order until `target_chars` is
    reached, so the slice is deterministic and re-creatable. Formatted exactly
    like wikitext2_test.txt: non-empty documents joined by newlines.
    """
    print("==> Downloading a C4 (en) validation shard")
    shard = hf_hub_download(
        repo_id="allenai/c4",
        filename="en/c4-validation.00000-of-00008.json.gz",
        repo_type="dataset",
    )
    parts, total, n_docs = [], 0, 0
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("text", "").strip()
            if not text:
                continue
            parts.append(text)
            total += len(text) + 1  # +1 for the joining newline
            n_docs += 1
            if total >= target_chars:
                break
    text = "\n".join(parts)
    out_path = out_dir / "c4_test.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"    wrote {out_path} ({len(text):,} characters from {n_docs:,} documents)")


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
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wikitext2", "instruction", "c4"],
        default=["wikitext2", "instruction"],
        help="Which corpora to build. Default is the original pair, so existing "
        "usage is unchanged; add 'c4' for the non-WikiText-overlapping "
        "perplexity corpus (a ~1.29 M char C4-en validation slice, sized to "
        "match wikitext2_test.txt).",
    )
    parser.add_argument("--c4-target-chars", type=int, default=DEFAULT_C4_TARGET_CHARS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if "wikitext2" in args.datasets:
        prepare_wikitext2(args.out_dir)
    if "c4" in args.datasets:
        prepare_c4(args.out_dir, args.c4_target_chars)
    if "instruction" in args.datasets:
        prepare_instruction_eval(args.out_dir, args.n_per_subject, args.subjects)


if __name__ == "__main__":
    main()

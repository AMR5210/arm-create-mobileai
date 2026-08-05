#!/usr/bin/env python3
"""Benchmark harness shared by every phase (fp16 baseline, PTQ-2bit, QAT-2bit).

Measures, for a single GGUF model:
  - disk size (bytes)
  - peak RAM of the llama-bench run (bytes)
  - prompt-processing and token-generation throughput (tokens/sec), via llama-bench
  - WikiText-2 perplexity, via llama-perplexity
  - instruction-following forced-choice accuracy, via scripts/instruction_eval.py

Writes one JSON record to results/<tag>.json. The --tag and --device flags exist
so results from different models and different hardware never get conflated --
per docs/METHODOLOGY.md, only iPhone 17 Pro Max (and optionally iPhone 12) runs are
the numbers that go in the submission; Mac mini runs are dev/sanity-check only.
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from instruction_eval import run_instruction_eval  # noqa: E402

PERPLEXITY_RE = None  # compiled lazily to avoid import at module scope ordering issues


def _perplexity_regex():
    import re

    global PERPLEXITY_RE
    if PERPLEXITY_RE is None:
        PERPLEXITY_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)")
    return PERPLEXITY_RE


def rss_bytes_of_last_child() -> int:
    """Peak RSS of the most recently reaped child process.

    Only accurate when called immediately after the FIRST subprocess a fresh
    Python process has spawned (RUSAGE_CHILDREN is a running max across all
    reaped children, not per-call) -- see caller ordering in main().
    """
    max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux reports KB, macOS/BSD reports bytes.
    multiplier = 1024 if platform.system() == "Linux" else 1
    return max_rss * multiplier


def run_llama_bench(llama_bench: Path, model_path: Path) -> tuple[dict, int]:
    cmd = [str(llama_bench), "-m", str(model_path), "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    peak_rss = rss_bytes_of_last_child()

    records = json.loads(result.stdout)
    pp_ts, tg_ts = [], []
    for rec in records:
        if rec.get("n_gen", 0) == 0 and rec.get("n_prompt", 0) > 0:
            pp_ts.append(rec["avg_ts"])
        elif rec.get("n_prompt", 0) == 0 and rec.get("n_gen", 0) > 0:
            tg_ts.append(rec["avg_ts"])

    summary = {
        "prompt_tokens_per_sec": sum(pp_ts) / len(pp_ts) if pp_ts else None,
        "gen_tokens_per_sec": sum(tg_ts) / len(tg_ts) if tg_ts else None,
        "raw": records,
    }
    return summary, peak_rss


def run_perplexity(llama_perplexity: Path, model_path: Path, corpus: Path) -> float | None:
    cmd = [str(llama_perplexity), "-m", str(model_path), "-f", str(corpus)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match = _perplexity_regex().search(result.stdout + result.stderr)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="e.g. baseline-fp16, ptq-2bit, qat-2bit")
    parser.add_argument("--device", required=True, help="e.g. 'iPhone 17 Pro Max', 'Mac mini M4 (dev)'")
    parser.add_argument("--model", type=Path, required=True, help="path to .gguf file")
    parser.add_argument("--llama-bin-dir", type=Path, default=Path("third_party/llama.cpp/build/bin"))
    parser.add_argument("--wikitext2", type=Path, default=Path("eval/data/wikitext2_test.txt"))
    parser.add_argument("--instruction-eval", type=Path, default=Path("eval/data/instruction_eval.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-instruction-eval", action="store_true")
    args = parser.parse_args()

    llama_bench = args.llama_bin_dir / "llama-bench"
    llama_perplexity = args.llama_bin_dir / "llama-perplexity"
    llama_cli = args.llama_bin_dir / "llama-cli"

    print(f"==> [{args.tag} / {args.device}] disk size")
    disk_bytes = args.model.stat().st_size

    print(f"==> [{args.tag} / {args.device}] llama-bench (speed + peak RAM)")
    speed, peak_rss = run_llama_bench(llama_bench, args.model)

    print(f"==> [{args.tag} / {args.device}] perplexity (wikitext2)")
    perplexity = run_perplexity(llama_perplexity, args.model, args.wikitext2)

    instruction_result = None
    if not args.skip_instruction_eval:
        print(f"==> [{args.tag} / {args.device}] instruction-following eval")
        instruction_result = run_instruction_eval(llama_cli, args.model, args.instruction_eval)

    record = {
        "tag": args.tag,
        "device": args.device,
        "model_path": str(args.model),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disk_bytes": disk_bytes,
        "peak_ram_bytes": peak_rss,
        "prompt_tokens_per_sec": speed["prompt_tokens_per_sec"],
        "gen_tokens_per_sec": speed["gen_tokens_per_sec"],
        "perplexity": perplexity,
        # Forced-choice accuracy over A/B/C/D is the reported instruction metric.
        # It does not depend on output formatting or generation budget, unlike the
        # generate-and-parse figures, and chance is a well-defined 25%. See
        # scripts/instruction_eval.py's module docstring.
        "instruction_forced_choice_accuracy": (
            instruction_result["forced_choice_accuracy"] if instruction_result else None
        ),
        "instruction_per_subject_forced_choice_accuracy": (
            instruction_result["per_subject_forced_choice_accuracy"] if instruction_result else None
        ),
        "llama_bench_raw": speed["raw"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(record, indent=2))

    print(f"\nWrote {out_path}")
    print(f"  disk:        {disk_bytes / 1e6:.1f} MB")
    print(f"  peak RAM:    {peak_rss / 1e6:.1f} MB")
    print(f"  prompt tok/s:{speed['prompt_tokens_per_sec']}")
    print(f"  gen tok/s:   {speed['gen_tokens_per_sec']}")
    print(f"  perplexity:  {perplexity}")
    if instruction_result:
        print(f"  forced-choice acc: {instruction_result['forced_choice_accuracy']:.1%}")


if __name__ == "__main__":
    main()

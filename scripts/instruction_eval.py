"""Instruction-following eval: runs each question in eval/data/instruction_eval.jsonl
through llama-cli as a plain-completion multiple-choice prompt and scores the
extracted answer letter against ground truth.

Not a rigorous log-likelihood multiple-choice scorer (llama.cpp's perplexity
binary supports that via a separate binary data format) -- this is a lightweight,
generation-based slice appropriate for a small hackathon eval set. Good enough to
show a clear PTQ-vs-QAT quality gap, not meant as a publishable MMLU number.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ANSWER_RE = re.compile(r"\b([ABCD])\b")
LETTERS = "ABCD"


def build_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "Answer the following multiple-choice question with only the letter of the "
        "correct choice.",
        "",
        f"Question: {question}",
    ]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def ask_model(llama_cli: Path, model_path: Path, prompt: str, timeout: int = 120) -> str:
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", prompt,
        "-n", "8",
        "--temp", "0",
        "-no-cnv",
        "--no-display-prompt",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout


def run_instruction_eval(llama_cli: Path, model_path: Path, eval_jsonl: Path) -> dict:
    rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]

    per_question = []
    correct = 0
    per_subject_correct: dict[str, int] = {}
    per_subject_total: dict[str, int] = {}

    for row in rows:
        prompt = build_prompt(row["question"], row["choices"])
        raw_output = ask_model(llama_cli, model_path, prompt)
        match = ANSWER_RE.search(raw_output)
        predicted_letter = match.group(1) if match else None
        predicted_idx = LETTERS.index(predicted_letter) if predicted_letter else -1
        is_correct = predicted_idx == row["answer"]

        correct += int(is_correct)
        subject = row.get("subject", "unknown")
        per_subject_total[subject] = per_subject_total.get(subject, 0) + 1
        per_subject_correct[subject] = per_subject_correct.get(subject, 0) + int(is_correct)

        per_question.append(
            {
                "subject": subject,
                "question": row["question"],
                "expected": LETTERS[row["answer"]],
                "predicted": predicted_letter,
                "correct": is_correct,
            }
        )

    n = len(rows)
    accuracy = correct / n if n else 0.0
    per_subject_accuracy = {
        subject: per_subject_correct[subject] / per_subject_total[subject]
        for subject in per_subject_total
    }

    return {
        "n_questions": n,
        "accuracy": accuracy,
        "per_subject_accuracy": per_subject_accuracy,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--eval-jsonl", type=Path, default=Path("eval") / "data" / "instruction_eval.jsonl"
    )
    args = parser.parse_args()

    result = run_instruction_eval(args.llama_cli, args.model, args.eval_jsonl)
    print(f"Accuracy: {result['accuracy']:.1%} ({result['n_questions']} questions)")
    for subject, acc in result["per_subject_accuracy"].items():
        print(f"  {subject}: {acc:.1%}")


if __name__ == "__main__":
    main()

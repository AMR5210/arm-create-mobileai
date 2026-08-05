"""Instruction-following eval over eval/data/instruction_eval.jsonl.

The reported metric is forced-choice accuracy: the model picks among exactly
A/B/C/D and every question yields a prediction, so chance is 25%.

This is implemented with a single-token GBNF grammar (`root ::= [ABCD]`) and
greedy sampling. Constraining the grammar masks every token except the four
answer labels, and greedy then selects the highest-scoring survivor -- which is
argmax over the A/B/C/D logits at the first generated position. The iOS harness
(ios/LlamaBench/LlamaBench/InstructionEval.swift) reads those logits directly;
the two are equivalent by construction and their per-question predictions are
compared to confirm it.

Forced choice supersedes a generate-and-parse approach whose result depended on
output formatting. Free generation left 37-73% of answers with no extractable
letter and placed every variant at or below chance
(results/logs/instr_eval_diag_summary.json), including the fp16 baseline. Parse
rate separated variants but tracked prompt-template conformance and shifted with
the token budget. Both remain available as diagnostics via
--include-free-generation.

Prompting note: llama-cli applies the model's chat template whenever the model
ships one, and `-no-cnv` does not suppress it. Qwen3's template defaults to its
reasoning mode, which spends the budget on a "<think>" preamble, so
--chat-template-kwargs disables thinking. Measured against llama-cli at the
checked-out revision.
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


ANSWER_GRAMMAR = "root ::= [ABCD]"


def ask_model_forced_choice(llama_cli: Path, model_path: Path, prompt: str,
                            timeout: int = 120) -> str | None:
    """One grammar-constrained token: the model's preferred answer label."""
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", prompt,
        "-n", "1",
        "--temp", "0",
        "--single-turn",
        "--grammar", ANSWER_GRAMMAR,
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--no-display-prompt",
        # Restricts stdout to the generation. llama-cli otherwise emits a startup
        # banner and log lines, which contain capitals in the A-D range.
        "--log-disable",
        "--simple-io",
        "--no-warmup",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None
    # llama-cli echoes the prompt even with --no-display-prompt, and truncates it
    # when long, so the echo cannot be located by a marker inside the prompt. The
    # stable property is the layout: generated text is the last non-empty line
    # before the "[ Prompt: ... ]" performance footer.
    #
    # The answer letter is read from that line alone, so that only generated text
    # is considered. A longer line indicates the expected layout does not hold and
    # is reported as unparseable.
    head = result.stdout.split("[ Prompt:")[0]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if len(last) > 4:
        return None
    for ch in last:
        if ch in LETTERS:
            return ch
    return None


def ask_model(llama_cli: Path, model_path: Path, prompt: str, timeout: int = 120) -> str:
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", prompt,
        "-n", "8",
        "--temp", "0",
        "-no-cnv",
        # llama-cli enables per-turn looping whenever the model ships a chat
        # template, which is the case for every model here, and `-no-cnv` alone
        # does not exit after the turn. --single-turn is required: without it the
        # process waits at an interactive prompt on a closed stdin until the
        # timeout elapses, returning "".
        "--single-turn",
        # Qwen3's chat template defaults to its reasoning ("thinking") mode, which
        # spends the whole `-n 8` budget on a "<think>" preamble without reaching a
        # letter. The template's own toggle disables it (see qat/data.py's chat
        # template handling), yielding a direct answer.
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--no-display-prompt",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout


def run_instruction_eval(llama_cli: Path, model_path: Path, eval_jsonl: Path,
                         include_free_generation: bool = False) -> dict:
    rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]
    n = len(rows)

    per_question = []
    fc_correct = 0
    fc_predictions = []
    fc_subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}

    # Diagnostics, only populated when the free-generation pass runs.
    gen_correct = 0
    gen_parsed = 0
    gen_subject_correct: dict[str, int] = {}
    gen_subject_parsed: dict[str, int] = {}

    for row in rows:
        prompt = build_prompt(row["question"], row["choices"])
        subject = row.get("subject", "unknown")
        subject_total[subject] = subject_total.get(subject, 0) + 1

        # --- reported metric: forced choice over A/B/C/D ---
        fc_letter = ask_model_forced_choice(llama_cli, model_path, prompt)
        fc_idx = LETTERS.index(fc_letter) if fc_letter else -1
        fc_is_correct = fc_idx == row["answer"]
        fc_correct += int(fc_is_correct)
        fc_predictions.append(fc_letter or "?")
        fc_subject_correct[subject] = fc_subject_correct.get(subject, 0) + int(fc_is_correct)

        entry = {
            "subject": subject,
            "question": row["question"],
            "expected": LETTERS[row["answer"]],
            "forced_choice_predicted": fc_letter,
            "forced_choice_correct": fc_is_correct,
        }

        # --- diagnostics: free generation, parsed for a letter ---
        if include_free_generation:
            raw_output = ask_model(llama_cli, model_path, prompt)
            completion = raw_output.rsplit("Answer:", 1)[-1]
            match = ANSWER_RE.search(completion)
            gen_letter = match.group(1) if match else None
            gen_idx = LETTERS.index(gen_letter) if gen_letter else -1
            gen_is_correct = gen_idx == row["answer"]
            gen_correct += int(gen_is_correct)
            gen_parsed += int(gen_letter is not None)
            gen_subject_correct[subject] = gen_subject_correct.get(subject, 0) + int(gen_is_correct)
            gen_subject_parsed[subject] = gen_subject_parsed.get(subject, 0) + int(gen_letter is not None)
            entry["generated_predicted"] = gen_letter
            entry["generated_correct"] = gen_is_correct

        per_question.append(entry)

    result = {
        "n_questions": n,
        "method": "forced choice: single-token grammar root ::= [ABCD], greedy "
                  "(equivalent to argmax over the A/B/C/D logits at the first "
                  "generated position)",
        "forced_choice_accuracy": fc_correct / n if n else 0.0,
        "per_subject_forced_choice_accuracy": {
            subject: fc_subject_correct[subject] / subject_total[subject]
            for subject in subject_total
        },
        "forced_choice_correct": fc_correct,
        "forced_choice_predictions": "".join(fc_predictions),
        "per_question": per_question,
    }

    if include_free_generation:
        result["parse_rate_diagnostic"] = gen_parsed / n if n else 0.0
        result["per_subject_parse_rate_diagnostic"] = {
            subject: gen_subject_parsed[subject] / subject_total[subject]
            for subject in subject_total
        }
        result["accuracy_diagnostic"] = gen_correct / n if n else 0.0
        result["per_subject_accuracy_diagnostic"] = {
            subject: gen_subject_correct[subject] / subject_total[subject]
            for subject in subject_total
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--eval-jsonl", type=Path, default=Path("eval") / "data" / "instruction_eval.jsonl"
    )
    parser.add_argument("--include-free-generation", action="store_true",
                        help="also run the generate-and-parse pass and report its parse rate "
                             "and accuracy as diagnostics (roughly doubles runtime)")
    args = parser.parse_args()

    result = run_instruction_eval(args.llama_cli, args.model, args.eval_jsonl,
                                  include_free_generation=args.include_free_generation)
    print(f"Forced-choice accuracy: {result['forced_choice_accuracy']:.1%} "
          f"({result['forced_choice_correct']}/{result['n_questions']}, chance 25%)")
    for subject, acc in result["per_subject_forced_choice_accuracy"].items():
        print(f"  {subject}: {acc:.1%}")
    print(f"predictions: {result['forced_choice_predictions']}")
    if args.include_free_generation:
        print(f"(diagnostic) parse rate: {result['parse_rate_diagnostic']:.1%}"
              f"  free-generation accuracy: {result['accuracy_diagnostic']:.1%}")


if __name__ == "__main__":
    main()

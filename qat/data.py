"""Loads and formats QAT fine-tuning data.

Up to three sources, blended (see build_blended_examples):
  - Alpaca instruction data (prompt masked out of the loss).
  - WikiText-2 *train* split, plain language-modeling text (every token
    supervised). Mixing the eval domain into training -- WikiText-2 is also the
    perplexity eval corpus -- closes the instruction-only-train / wikitext-eval
    domain mismatch. Recent low-bit-QAT work on Qwen3 reports meaningful
    perplexity gains from aligning the QAT data with the eval domain rather than
    training on instructions alone.
  - FineWeb, a general web-text corpus (plain LM, every token supervised),
    optional and off by default. Adds domain breadth beyond WikiText-2/Alpaca
    without reintroducing the leakage this project already fixed once: it must
    NOT overlap WikiText-2 (Wikipedia-derived) or C4 (results/c4_perplexity.json's
    held-out generalization check). RedPajama/SlimPajama were considered first
    but both are unusable here -- RedPajama-Data-1T is a legacy loading-script
    dataset (datasets>=5 dropped script execution entirely) and SlimPajama-627B
    is no longer accessible on the Hub under its published name. Both are also
    mixtures that literally include C4 and Wikipedia as named sub-components,
    so using them would have required carefully filtering those back out.
    FineWeb (arXiv:2406.17557) is a cleaner fit: pure deduplicated CommonCrawl
    text, no Wikipedia or C4 component at all, so it can't overlap either
    corpus by construction, and it loads directly (parquet, no trust_remote_code).
"""
import random

import torch
from datasets import load_dataset

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(example: dict) -> str:
    if example.get("input"):
        return ALPACA_PROMPT_WITH_INPUT.format(instruction=example["instruction"], input=example["input"])
    return ALPACA_PROMPT_NO_INPUT.format(instruction=example["instruction"])


def load_alpaca_examples(max_examples: int = 2000, seed: int = 0, dataset_name: str = "tatsu-lab/alpaca"):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(max_examples, len(ds))))


def load_wikitext2_train_examples(
    tokenizer,
    max_length: int = 512,
    max_examples: int = 2000,
    seed: int = 0,
    dataset_name: str = "Salesforce/wikitext",
    config: str = "wikitext-2-raw-v1",
) -> list[dict]:
    """Plain language-modeling examples from the WikiText-2 *train* split.

    The whole corpus is concatenated and cut into contiguous ``max_length``
    token chunks; every token is a training target (labels = input_ids, nothing
    masked), unlike the Alpaca examples whose prompt is masked out. Uses the
    TRAIN split, never the test split that perplexity is evaluated on, so there
    is no train/eval leakage.
    """
    # Namespaced repo id ("Salesforce/wikitext"): newer huggingface_hub (1.x)
    # rejects the legacy canonical id "wikitext" (must be "namespace/name"),
    # same fix as scripts/prepare_eval_data.py. Override with --wikitext-dataset.
    ds = load_dataset(dataset_name, config, split="train")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    ids = tokenizer(text)["input_ids"]

    examples = []
    for start in range(0, len(ids) - 1, max_length):
        chunk = ids[start : start + max_length]
        if len(chunk) < 2:  # need at least one (input, next-token) pair
            continue
        examples.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "labels": list(chunk),  # plain LM: supervise every token
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def load_fineweb_train_examples(
    tokenizer,
    max_length: int = 512,
    max_examples: int = 2000,
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "sample-10BT",
) -> list[dict]:
    """Plain language-modeling examples from FineWeb (arXiv:2406.17557), a
    deduplicated CommonCrawl web-text corpus -- see the module docstring for
    why this is the general-domain corpus used here instead of RedPajama/
    SlimPajama.

    Streamed rather than downloaded whole (``sample-10BT`` alone is ~10B
    tokens): documents are pulled in dataset order and accumulated until
    there's roughly enough raw text for ``max_examples`` chunks of
    ``max_length`` tokens (a ~4 chars/token heuristic, generous enough that
    tokenizing rarely comes up short), then tokenized and cut into
    contiguous ``max_length`` chunks exactly like
    ``load_wikitext2_train_examples`` -- every token is a training target
    (labels = input_ids, nothing masked).
    """
    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    target_chars = max_examples * max_length * 4
    parts, total_chars = [], 0
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        parts.append(text)
        total_chars += len(text) + 1  # +1 for the joining newline
        if total_chars >= target_chars:
            break
    text = "\n".join(parts)
    ids = tokenizer(text)["input_ids"]

    examples = []
    for start in range(0, len(ids) - 1, max_length):
        chunk = ids[start : start + max_length]
        if len(chunk) < 2:  # need at least one (input, next-token) pair
            continue
        examples.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "labels": list(chunk),  # plain LM: supervise every token
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def blend_target_counts(n_alpaca: int, fracs: dict[str, float]) -> dict[str, int]:
    """How many examples each named corpus needs so that, blended with a
    fixed ``n_alpaca`` Alpaca examples, it lands at its target fraction of
    the FINAL blend -- e.g. ``{"wikitext2": 0.5, "fineweb": 0.2}`` means the
    final mix is ~50% WikiText-2 / ~20% FineWeb / ~30% Alpaca. Fractions are
    shares of the same total, not of each other, so order doesn't matter.

    This is the count each corpus loader should be asked to PRODUCE (so a
    3-way blend doesn't silently cap a source at whatever a 1- or 2-way
    blend needed); build_blended_examples then re-derives the same counts to
    select from the (possibly smaller, if a corpus ran short) loaded lists,
    so the two stay consistent by construction.
    """
    active = {name: frac for name, frac in fracs.items() if frac > 0}
    counts = {name: 0 for name in fracs}
    if not active:
        return counts
    total_frac = sum(active.values())
    if total_frac >= 1:
        raise ValueError(f"corpus fractions must sum to less than 1 (got {total_frac})")
    total = n_alpaca / (1 - total_frac)
    counts.update({name: round(total * frac) for name, frac in active.items()})
    return counts


def build_blended_examples(
    alpaca_examples: list[dict],
    extra_corpora: list[tuple[str, list[dict], float]],
    seed: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """Blend Alpaca instruction examples with zero or more plain-LM corpora
    (WikiText-2, FineWeb, ...), each targeting a fraction of the FINAL blended
    set (see blend_target_counts), then shuffle everything together.

    ``extra_corpora`` is a list of ``(name, examples, target_frac)`` triples.
    Alpaca always keeps every example; each other corpus contributes up to
    its target count, capped by how many examples it actually loaded (in
    which case the realized fraction -- returned per name -- is lower than
    requested; callers should size their loader calls from
    blend_target_counts to make that cap the exception, not the norm).

    Returns ``(blended_examples, {name: n_taken})``.
    """
    n_alpaca = len(alpaca_examples)
    active = [(name, examples, frac) for name, examples, frac in extra_corpora if frac > 0 and examples]
    if not active:
        return list(alpaca_examples), {name: 0 for name, _, _ in extra_corpora}

    target_counts = blend_target_counts(n_alpaca, {name: frac for name, _, frac in active})

    blended = list(alpaca_examples)
    taken_counts = {}
    for name, examples, _ in active:
        taken = list(examples[: target_counts[name]])
        blended.extend(taken)
        taken_counts[name] = len(taken)
    for name, _, _ in extra_corpora:
        taken_counts.setdefault(name, 0)

    random.Random(seed).shuffle(blended)
    return blended, taken_counts


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the matching prefix of two token-id sequences.

    Tokenizing the prompt alone is not guaranteed to be an exact prefix of
    tokenizing prompt+response together -- BPE-style tokenizers can merge
    differently right at the boundary. Using the actual common prefix (rather
    than just trusting len(tokenized_prompt)) keeps the prompt/response split
    correct regardless of where such a boundary mismatch occurs.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def build_supervised_example(example: dict, tokenizer, max_length: int = 512) -> dict:
    """Tokenizes prompt+response, masking the prompt portion out of the loss
    (label = -100) so the model is only trained to produce the response.
    """
    prompt = format_prompt(example)
    response = example["output"] + tokenizer.eos_token
    full_text = prompt + response

    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length)
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length)

    input_ids = tokenized_full["input_ids"]
    labels = list(input_ids)
    prompt_len = _common_prefix_len(input_ids, tokenized_prompt["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


def collate_fn(batch: list[dict], tokenizer) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id

    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

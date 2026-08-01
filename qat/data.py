"""Loads and formats QAT fine-tuning data.

Two sources, blended (see build_blended_examples):
  - Alpaca instruction data (prompt masked out of the loss).
  - WikiText-2 *train* split, plain language-modeling text (every token
    supervised). Mixing the eval domain into training -- WikiText-2 is also the
    perplexity eval corpus -- closes the instruction-only-train / wikitext-eval
    domain mismatch. Recent low-bit-QAT work on Qwen3 reports meaningful
    perplexity gains from aligning the QAT data with the eval domain rather than
    training on instructions alone.
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
    dataset_name: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
) -> list[dict]:
    """Plain language-modeling examples from the WikiText-2 *train* split.

    The whole corpus is concatenated and cut into contiguous ``max_length``
    token chunks; every token is a training target (labels = input_ids, nothing
    masked), unlike the Alpaca examples whose prompt is masked out. Uses the
    TRAIN split, never the test split that perplexity is evaluated on, so there
    is no train/eval leakage.
    """
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


def build_blended_examples(alpaca_examples: list[dict], wiki_examples: list[dict],
                           wikitext_frac: float, seed: int = 0) -> list[dict]:
    """Blend supervised Alpaca and WikiText-2 examples so that a ``wikitext_frac``
    fraction of the returned examples are WikiText-2, then shuffle them together.

    Keeps all Alpaca examples and takes as many WikiText-2 examples as the ratio
    calls for (capped by how many are available); if WikiText-2 is the limiting
    side, the realized fraction is reported by the caller.
    """
    n_alpaca = len(alpaca_examples)
    if wikitext_frac <= 0 or not wiki_examples:
        return list(alpaca_examples)
    if wikitext_frac >= 1:
        return list(wiki_examples)
    # want n_wiki / (n_alpaca + n_wiki) == wikitext_frac
    n_wiki_target = round(n_alpaca * wikitext_frac / (1 - wikitext_frac))
    wiki = list(wiki_examples[:n_wiki_target])
    blended = list(alpaca_examples) + wiki
    random.Random(seed).shuffle(blended)
    return blended


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

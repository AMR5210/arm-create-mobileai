"""Loads and formats the Alpaca instruction dataset for the QAT fine-tune."""
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
    prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels))
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

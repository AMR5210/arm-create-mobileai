"""In-training WikiText-2 perplexity monitoring for the live PyTorch model
(before any GGUF export exists). Uses the standard HF sliding-window
perplexity recipe.

This is for watching the training trend only, optionally over a truncated
slice for speed (--eval-max-tokens in scripts/train_qat.py). The perplexity
number that actually goes in the submission comes from llama-perplexity on
the exported GGUF (scripts/benchmark.py), over the full corpus, matching how
the PTQ and baseline variants are measured.
"""
import torch


def compute_perplexity(
    model,
    tokenizer,
    text_path,
    device,
    max_length: int = 1024,
    stride: int = 512,
    max_eval_tokens: int | None = None,
) -> float:
    text = open(text_path, encoding="utf-8").read()
    if max_eval_tokens is not None:
        # Cheap character-level pre-truncation before tokenizing; exact token
        # count is trimmed again below, this just avoids tokenizing the
        # entire corpus when only a fast progress check is needed.
        text = text[: max_eval_tokens * 8]

    was_training = model.training
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids
    if max_eval_tokens is not None:
        input_ids_full = input_ids_full[:, :max_eval_tokens]
    seq_len = input_ids_full.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = input_ids_full[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break

    if was_training:
        model.train()

    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()

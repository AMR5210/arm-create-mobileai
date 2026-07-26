#!/usr/bin/env python3
"""QAT fine-tuning entrypoint.

Wraps the base model's linear layers with fake-quantization (initialized
from an INT4 rounding, trained toward a 2-bit target -- the plan's
progressive FP16 -> INT4 -> INT2 route), then does a short instruction
fine-tune on a small Alpaca slice with a straight-through estimator, printing
WikiText-2 perplexity periodically so the trend can be checked before
extending training time.

This runs on the GPU cloud instance -- not this repo's dev sandbox, and not
the Mac mini. After training, scripts/export_qat_gguf.sh converts the saved
checkpoint into the same GGUF Q2_K format as the PTQ-2bit baseline.
"""
import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from qat.apply_qat import apply_qat, fake_quantized_state_dict, materialize_qat  # noqa: E402
from qat.data import build_supervised_example, collate_fn, load_alpaca_examples  # noqa: E402
from qat.eval_utils import compute_perplexity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("models/qwen3-0.6b-hf"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen3-0.6b-qat-hf"))
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--init-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--wikitext2", type=Path, default=Path("eval/data/wikitext2_test.txt"))
    parser.add_argument(
        "--eval-max-tokens",
        type=int,
        default=20000,
        help="Truncates wikitext2 for fast mid-training checks. The number "
        "reported in the submission comes from llama-perplexity on the "
        "exported GGUF, over the full corpus.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Every N steps, saves a non-destructive safety-net checkpoint "
        "(weights only, no optimizer state) to --checkpoint-dir, so an "
        "interrupted session (e.g. a Colab disconnect) doesn't lose all "
        "training progress. Set to 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Defaults to <output-dir>-partial.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. QAT's loss surface is noisier than "
        "ordinary fine-tuning (every forward pass injects quantization "
        "rounding noise), so an unclipped occasional large gradient can "
        "throw a weight group into a badly-scaled region and cascade to "
        "NaN within tens of steps.",
    )
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir or args.output_dir.parent / f"{args.output_dir.name}-partial"

    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"==> Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)

    print(
        f"==> Wrapping linear layers with fake-quantization "
        f"(target {args.bits}-bit, initialized from {args.init_bits}-bit rounding, "
        f"group size {args.group_size})"
    )
    replaced = apply_qat(model, bits=args.bits, group_size=args.group_size, init_bits=args.init_bits)
    print(f"    wrapped {len(replaced)} linear layers")

    print(f"==> Loading {args.max_examples} examples from {args.dataset}")
    raw_examples = load_alpaca_examples(max_examples=args.max_examples, dataset_name=args.dataset)
    examples = [build_supervised_example(ex, tokenizer, args.max_length) for ex in raw_examples]

    dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if args.save_every:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).config.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"==> Safety-net checkpoints will be written to {checkpoint_dir} every {args.save_every} steps")

    print("==> WikiText-2 perplexity right after fake-quant wrapping (pre fine-tune)")
    ppl = compute_perplexity(model, tokenizer, args.wikitext2, device, max_eval_tokens=args.eval_max_tokens)
    print(f"    perplexity: {ppl:.4f}")

    step = 0
    model.train()
    done = False
    while not done:
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss):
                # Don't let a single bad batch permanently poison AdamW's
                # per-parameter moment buffers with NaN/Inf (which, once
                # NaN, stay NaN for the rest of training). Skip the update
                # and move on instead -- no point even backpropagating a
                # non-finite loss.
                print(f"    [warning] step {step + 1}: non-finite loss ({loss.item()}), skipping this update")
                optimizer.zero_grad()
                continue

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.eval_every == 0 or step == args.max_steps:
                ppl = compute_perplexity(
                    model, tokenizer, args.wikitext2, device, max_eval_tokens=args.eval_max_tokens
                )
                print(f"step {step}/{args.max_steps}  loss={loss.item():.4f}  wikitext2_ppl={ppl:.4f}")

            if args.save_every and step % args.save_every == 0:
                state_dict = fake_quantized_state_dict(accelerator.unwrap_model(model))
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                print(f"    [checkpoint] step {step}: safety-net weights saved to {checkpoint_dir}")

            if step >= args.max_steps:
                done = True
                break

    print("==> Materializing fake-quantized weights into plain Linear layers for export")
    unwrapped = accelerator.unwrap_model(model)
    materialize_qat(unwrapped)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Saving HF checkpoint to {args.output_dir}")
    unwrapped.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(
        "==> Done. Next: scripts/export_qat_gguf.sh to convert this checkpoint into "
        "GGUF Q2_K for the final benchmark against ptq-2bit."
    )


if __name__ == "__main__":
    main()

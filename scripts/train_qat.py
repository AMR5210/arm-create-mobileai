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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

sys.path.insert(0, str(Path(__file__).parent.parent))
from qat.apply_qat import apply_qat, fake_quantized_state_dict, materialize_qat, set_qat_bits  # noqa: E402
from qat.attn_backend import safe_attn_implementation  # noqa: E402
from qat.calibrate_clip import calibrate_asymmetric_clipping  # noqa: E402
from qat.data import build_supervised_example, collate_fn, load_alpaca_examples  # noqa: E402
from qat.eval_utils import compute_perplexity  # noqa: E402


def scheduled_bits(step: int, warmup_bits: int, target_bits: int, anneal_steps: int) -> int:
    """Forward bit-width for a given step: linearly step down from
    warmup_bits to target_bits over anneal_steps, then hold at target.
    """
    if warmup_bits <= target_bits or step >= anneal_steps or anneal_steps <= 0:
        return target_bits
    frac = step / anneal_steps
    return max(target_bits, round(warmup_bits - frac * (warmup_bits - target_bits)))


def _kl_response_only(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) = sum_c p*(log p - log q), masked mean over response tokens."""
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return kl[mask].mean()


def cakld_loss(student_logits, teacher_logits, labels, gamma: float):
    """BitDistiller's Confidence-Aware KLD (arXiv:2402.10631, Eq. 5).

    Blends reverse KL (mode-seeking, weight gamma) and forward KL
    (mode-covering, weight 1-gamma), response tokens only (same -100 mask as
    the hard label loss):

        D_CAKLD(P_T || P_S) = gamma * KL(P_S || P_T) + (1-gamma) * KL(P_T || P_S)

    `gamma` is a single scalar precomputed once, prior to training, as the
    teacher's average confidence on the actual next token (see
    precompute_cakld_gamma) -- NOT a per-token weight. When the teacher is
    confident on the training data, CAKLD leans mode-seeking; when it's
    uncertain, it leans mode-covering.
    """
    shift_student = student_logits[..., :-1, :].float()
    shift_teacher = teacher_logits[..., :-1, :].float()
    mask = labels[..., 1:] != -100
    if not mask.any():
        return shift_student.new_zeros(())
    student_log_probs = torch.log_softmax(shift_student, dim=-1)
    teacher_log_probs = torch.log_softmax(shift_teacher, dim=-1)
    reverse_kl = _kl_response_only(student_log_probs, teacher_log_probs, mask)  # KL(S || T)
    forward_kl = _kl_response_only(teacher_log_probs, student_log_probs, mask)  # KL(T || S)
    return gamma * reverse_kl + (1 - gamma) * forward_kl


@torch.no_grad()
def precompute_cakld_gamma(teacher_model, dataloader, device, n_batches: int = 10) -> float:
    """BitDistiller Eq. 5's gamma: the teacher's average probability on the
    actual next token, over a handful of calibration batches (Appendix
    A.2: ten batches, forward-only, no parameter updates), computed once
    before training starts. Response tokens only, matching this repo's
    -100 label mask.
    """
    teacher_model.eval()
    prob_sum = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].float()
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.clamp(min=0)  # -100 would break gather; masked out below anyway
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        prob_sum += token_log_probs.exp()[mask].sum().item()
        count += mask.sum().item()
    return prob_sum / count if count else 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("models/qwen3-0.6b-hf"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen3-0.6b-qat-hf"))
    parser.add_argument("--bits", type=int, default=2, help="Final target bit-width.")
    parser.add_argument("--init-bits", type=int, default=4)
    parser.add_argument(
        "--group-size", type=int, default=16,
        help="Per-group size for weight fake-quant. Default 16 aligns with the "
        "GGUF Q2_K sub-block size (16 elements), so the QAT grouping matches the "
        "deployment quantiser's granularity (see scripts/export_qat_gguf.py). "
        "Earlier runs used 32; 16 is the export-aligned default going forward.",
    )
    parser.add_argument(
        "--skip-layers",
        nargs="+",
        default=[],
        help="Substring patterns of layers to KEEP at full precision "
        "(mixed-precision QAT), on top of the always-skipped embeddings / "
        "lm_head / norms. Use for quantization-sensitive outlier-heavy layers "
        "that destabilize training -- run scripts/diagnose_qat_grads.py to see "
        "each layer's weight dynamic range and pick them. Example: "
        "--skip-layers layers.0.self_attn.k_proj",
    )
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--warmup-bits",
        type=int,
        default=4,
        help="Bit-width the forward pass STARTS at, annealed down to --bits "
        "over --bit-anneal-steps. Progressive quantization (the plan's "
        "FP16->INT4->INT2 route): a pretrained model quantized straight to "
        "2 bits perturbs the forward enough to explode gradients through the "
        "network's depth on step 1, before any learning happens. Starting at "
        "4 bits (near-lossless for these models) keeps the initial forward "
        "stable, then the perturbation is increased gradually while the "
        "weights adapt to track it. Set equal to --bits to disable annealing.",
    )
    parser.add_argument(
        "--bit-anneal-steps",
        type=int,
        default=150,
        help="Number of steps over which the forward bit-width steps down "
        "from --warmup-bits to --bits.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Linear LR warmup steps (0 -> --lr). Standard QAT stabilizer: "
        "avoids a large first update on the noisy quantized loss surface.",
    )
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
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=20,
        help="Abort if this many updates in a row are skipped for a "
        "non-finite loss/gradient. Once AdamW's moment buffers or the "
        "model's own parameters go NaN, every future batch produces NaN "
        "too, regardless of content -- skipping forever would just spin "
        "without making progress. Failing loudly here is better than a "
        "silent infinite loop burning GPU time.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for activation memory (recompute activations in "
        "the backward pass instead of storing them). Needed to fit the fp32 "
        "forward/backward on a small GPU; does not change the numerics, only "
        "when activations are materialized.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Weight for a knowledge-distillation term against a frozen teacher "
        "(the unmodified --base-model checkpoint, loaded separately from the QAT "
        "student). total_loss = hard_label_loss + distill_weight * CAKLD(teacher, "
        "student), computed only over response tokens (same -100 label mask as the "
        "hard loss). CAKLD (arXiv:2402.10631 Eq. 5) blends reverse and forward KL "
        "with a coefficient auto-estimated from the teacher's confidence on the "
        "training data, instead of a fixed KL direction. 0 (default) disables it "
        "entirely -- no teacher model is loaded and behavior is unchanged.",
    )
    parser.add_argument(
        "--cakld-calib-batches",
        type=int,
        default=10,
        help="Number of forward-only batches used to precompute CAKLD's gamma "
        "coefficient (arXiv:2402.10631 Appendix A.2 uses ten). Only used when "
        "--distill-weight > 0.",
    )
    parser.add_argument(
        "--calibrate-clip",
        action="store_true",
        help="Asymmetric clipping calibration (arXiv:2402.10631 Section 3.1, Eq. 3) "
        "before QAT starts: for each quantized layer, grid-search clip bounds "
        "(alpha, beta) that minimize output reconstruction error on real "
        "calibration activations, then permanently clamp that layer's raw weight "
        "to those bounds. Targets the same outlier-heavy layers --skip-layers "
        "works around, but at the quantization-scheme level (bounding the "
        "asymmetric range every group quantizes from) rather than excluding them.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adafactor", "sgd"],
        default="adamw",
        help="Update rule. Default 'adamw' is what the reference GPU run uses. "
        "'adafactor' / 'sgd' exist ONLY to shrink optimizer-state memory so a "
        "full-precision (fp32) run fits on a small GPU: AdamW keeps two fp32 "
        "moment buffers per weight (~2x model size), which alone can exceed a "
        "6 GB card for a 0.6B model. These alternatives keep the fp32 "
        "fake-quant forward/backward math IDENTICAL -- so a NaN/Inf originating "
        "there (the failure mode this project guards against) still surfaces in "
        "the loss / grad-norm checks regardless of optimizer -- they only "
        "change how the (finite) gradient is turned into a weight update.",
    )
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir or args.output_dir.parent / f"{args.output_dir.name}-partial"

    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"==> Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, attn_implementation=safe_attn_implementation()
    ).to(device)

    print(f"==> Loading {args.max_examples} examples from {args.dataset}")
    raw_examples = load_alpaca_examples(max_examples=args.max_examples, dataset_name=args.dataset)
    examples = [build_supervised_example(ex, tokenizer, args.max_length) for ex in raw_examples]

    # If an example's instruction+input is long enough that truncating
    # prompt+response to --max-length cuts off the response entirely, every
    # label ends up masked (-100) with nothing left to supervise. Cross-entropy
    # with ignore_index=-100 over zero unmasked tokens is 0/0 = NaN, regardless
    # of anything happening in the model -- drop those examples rather than
    # let them silently produce a NaN loss mid-training.
    n_before = len(examples)
    examples = [ex for ex in examples if any(label != -100 for label in ex["labels"])]
    if len(examples) < n_before:
        print(
            f"    dropped {n_before - len(examples)} example(s) whose response was fully "
            f"truncated away by --max-length {args.max_length} (no supervision signal left)"
        )

    dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    if args.calibrate_clip:
        print("==> Asymmetric clipping calibration (BitDistiller Eq. 3), before fake-quant wrapping")
        calib_batches = [batch for _, batch in zip(range(8), dataloader)]
        clipped = calibrate_asymmetric_clipping(
            model, calib_batches, bits=args.bits, group_size=args.group_size,
            device=device, extra_skip_patterns=tuple(args.skip_layers),
        )
        print(f"    calibrated and clamped {len(clipped)} layers")

    print(
        f"==> Wrapping linear layers with fake-quantization "
        f"(target {args.bits}-bit, initialized from {args.init_bits}-bit rounding, "
        f"group size {args.group_size})"
    )
    replaced = apply_qat(
        model, bits=args.bits, group_size=args.group_size, init_bits=args.init_bits,
        extra_skip_patterns=tuple(args.skip_layers),
    )
    print(f"    wrapped {len(replaced)} linear layers")
    if args.skip_layers:
        print(f"    kept at full precision (mixed-precision): {', '.join(args.skip_layers)}")

    teacher_model = None
    cakld_gamma = None
    if args.distill_weight > 0:
        print(f"==> Loading frozen teacher from {args.base_model} (distill_weight={args.distill_weight})")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.float32, attn_implementation=safe_attn_implementation()
        ).to(device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        cakld_gamma = precompute_cakld_gamma(teacher_model, dataloader, device, n_batches=args.cakld_calib_batches)
        print(f"    CAKLD gamma (teacher confidence on response tokens): {cakld_gamma:.4f}")

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print("==> Gradient checkpointing enabled (lower activation memory, same numerics)")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:  # adafactor -- factored 2nd moment, no 1st moment => tiny state
        from transformers.optimization import Adafactor

        optimizer = Adafactor(
            model.parameters(), lr=args.lr,
            scale_parameter=False, relative_step=False, warmup_init=False,
        )
    if args.optimizer != "adamw":
        print(f"==> Optimizer: {args.optimizer} (memory-reduced; fp32 fake-quant math unchanged)")
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    if args.warmup_bits > args.bits:
        print(
            f"==> Progressive bit-width: forward starts at {args.warmup_bits}-bit, "
            f"annealing to {args.bits}-bit over {args.bit_anneal_steps} steps"
        )

    if args.save_every:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).config.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"==> Safety-net checkpoints will be written to {checkpoint_dir} every {args.save_every} steps")

    print("==> WikiText-2 perplexity right after fake-quant wrapping (pre fine-tune)")
    ppl = compute_perplexity(model, tokenizer, args.wikitext2, device, max_eval_tokens=args.eval_max_tokens)
    print(f"    perplexity: {ppl:.4f}")

    step = 0
    consecutive_skips = 0
    model.train()
    done = False
    while not done:
        for batch in dataloader:
            current_bits = scheduled_bits(
                step, args.warmup_bits, args.bits, args.bit_anneal_steps
            )
            set_qat_bits(model, current_bits)

            outputs = model(**batch)
            hard_loss = outputs.loss
            kd_loss = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    )
                kd_loss = cakld_loss(outputs.logits, teacher_outputs.logits, batch["labels"], cakld_gamma)
                loss = hard_loss + args.distill_weight * kd_loss
            else:
                loss = hard_loss

            updated = False

            if not torch.isfinite(loss):
                # Don't let a single bad batch permanently poison AdamW's
                # per-parameter moment buffers with NaN/Inf (which, once
                # NaN, stay NaN for the rest of training). Skip the update
                # and move on instead -- no point even backpropagating a
                # non-finite loss.
                print(f"    [warning] step {step + 1}: non-finite loss ({loss.item()}), skipping this update")
                optimizer.zero_grad()
                consecutive_skips += 1
            else:
                accelerator.backward(loss)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # A finite loss does NOT guarantee a finite gradient -- backward()
                # can produce NaN/Inf gradients from a forward pass that looked
                # completely fine. clip_grad_norm_ does not sanitize this: clipping
                # a NaN-normed gradient with a NaN coefficient still yields NaN,
                # which optimizer.step() would then bake permanently into AdamW's
                # moment buffers. Check it explicitly instead of trusting the loss
                # check alone.
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    print(f"    [warning] step {step + 1}: non-finite grad norm ({grad_norm}), skipping this update")
                    optimizer.zero_grad()
                    consecutive_skips += 1
                else:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    consecutive_skips = 0
                    updated = True

            if consecutive_skips >= args.max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive updates skipped for non-finite "
                    f"loss/gradient at step {step} -- the model is very likely "
                    f"permanently NaN-poisoned at this point (once AdamW's moment "
                    f"buffers or the parameters themselves go NaN, they never "
                    f"recover, so every batch produces NaN forever regardless of "
                    f"content). Restart training from scratch, or from the last "
                    f"safety-net checkpoint in {checkpoint_dir} if step >= "
                    f"--save-every once. Consider a lower --lr and/or "
                    f"--max-grad-norm before retrying."
                )

            if not updated:
                continue

            if step % args.eval_every == 0 or step == args.max_steps:
                ppl = compute_perplexity(
                    model, tokenizer, args.wikitext2, device, max_eval_tokens=args.eval_max_tokens
                )
                if kd_loss is not None:
                    print(
                        f"step {step}/{args.max_steps}  loss={loss.item():.4f}  "
                        f"hard_loss={hard_loss.item():.4f}  kd_loss={kd_loss.item():.4f}  "
                        f"wikitext2_ppl={ppl:.4f}  bits={current_bits}  "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}  grad_norm={grad_norm:.3f}"
                    )
                else:
                    print(
                        f"step {step}/{args.max_steps}  loss={loss.item():.4f}  "
                        f"wikitext2_ppl={ppl:.4f}  bits={current_bits}  "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}  grad_norm={grad_norm:.3f}"
                    )

            if args.save_every and step % args.save_every == 0:
                state_dict = fake_quantized_state_dict(accelerator.unwrap_model(model))
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                print(f"    [checkpoint] step {step}: safety-net weights saved to {checkpoint_dir}")

            if step >= args.max_steps:
                done = True
                break

    # Make sure export bakes in the FINAL target bit-width, not whatever the
    # last training step happened to be at (they match once training runs past
    # --bit-anneal-steps, but force it so a short run truncated mid-anneal still
    # exports at the intended --bits rather than an intermediate width).
    set_qat_bits(model, args.bits)
    print(f"==> Materializing fake-quantized weights ({args.bits}-bit) into plain Linear layers for export")
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

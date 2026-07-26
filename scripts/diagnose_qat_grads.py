#!/usr/bin/env python3
"""Diagnostic: per-layer gradient magnitudes for a QAT-wrapped model, BEFORE
any clipping, on a few real batches.

Written to answer "where do the exploding/non-finite gradients come from?".
It reports, per parameter tensor:
  - the raw grad L2 norm (and whether it is inf/nan)
  - the max absolute grad element (the thing that overflows clip_grad_norm_'s
    fp32 sum-of-squares once it exceeds ~1e19)
  - the same at several candidate bit-widths, so you can see how much the
    instability is driven by the aggressiveness of the 2-bit target vs. the
    model itself.

Run this on the GPU box with the real model when a training run is skipping
every step for non-finite grad norm. It does NOT update any weights.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from qat.apply_qat import apply_qat, set_qat_bits, weight_dynamic_range  # noqa: E402
from qat.attn_backend import safe_attn_implementation  # noqa: E402
from qat.data import build_supervised_example, collate_fn, load_alpaca_examples  # noqa: E402


def grad_report(model) -> None:
    worst = []
    n_inf = n_nan = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        gmax = g.abs().max().item()
        # compute the norm in float64 so the report itself never overflows,
        # letting us distinguish "genuinely inf gradient" from "finite but
        # large enough that fp32 clip_grad_norm_ would overflow squaring it".
        gnorm = g.double().norm(2).item()
        is_inf = torch.isinf(g).any().item()
        is_nan = torch.isnan(g).any().item()
        n_inf += int(is_inf)
        n_nan += int(is_nan)
        worst.append((gnorm, gmax, name, is_inf, is_nan))

    worst.sort(reverse=True)
    print(f"  params with inf grad: {n_inf}, with nan grad: {n_nan}")
    print(f"  {'grad_L2(fp64)':>14}  {'max|grad|':>12}  fp32_norm_would_overflow  layer")
    for gnorm, gmax, name, is_inf, is_nan in worst[:15]:
        # fp32 sum-of-squares overflows ~ when max element^2 * numel > 3.4e38
        overflow = gmax > 1.8e19
        flag = "INF" if is_inf else ("NAN" if is_nan else ("yes" if overflow else "no"))
        print(f"  {gnorm:14.3e}  {gmax:12.3e}  {flag:>22}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("models/qwen3-0.6b-hf"))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--init-bits", type=int, default=4)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument(
        "--bits-to-probe",
        type=int,
        nargs="+",
        default=[4, 3, 2],
        help="Report per-layer grads at each of these forward bit-widths.",
    )
    parser.add_argument(
        "--trace-nan",
        action="store_true",
        help="Pinpoint where NaN/Inf first appears. Installs per-module forward "
        "and backward hooks that report the first module producing a non-finite "
        "activation (forward) or gradient (backward), and enables "
        "torch.autograd.set_detect_anomaly so the backward raises at the exact "
        "op that created the offending value, with a traceback into the forward.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, attn_implementation=safe_attn_implementation()
    ).to(device)
    apply_qat(model, bits=args.bits_to_probe[0], group_size=args.group_size, init_bits=args.init_bits)

    raw = load_alpaca_examples(max_examples=args.batch_size * args.n_batches, dataset_name=args.dataset)
    examples = [build_supervised_example(ex, tokenizer, args.max_length) for ex in raw]
    examples = [ex for ex in examples if any(l != -100 for l in ex["labels"])]
    dataloader = DataLoader(
        examples, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    print("\n===== weight dynamic range (most outlier-heavy first) =====")
    print("  Layers with a very large max/median are quantization-sensitive and")
    print("  candidates for --skip-layers (kept full-precision) in train_qat.py.")
    print(f"  {'max|w|':>11}  {'min_nonzero':>12}  {'max/median':>12}  layer")
    for name, max_abs, min_nz, max_over_median in weight_dynamic_range(model)[:12]:
        print(f"  {max_abs:11.3e}  {min_nz:12.3e}  {max_over_median:12.3e}  {name}")

    if args.trace_nan:
        install_nan_hooks(model)

    model.train()
    for bits in args.bits_to_probe:
        set_qat_bits(model, bits)
        print(f"\n===== forward bit-width = {bits} =====")
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)

        if args.trace_nan:
            _seen.clear()
            with torch.autograd.set_detect_anomaly(True):
                loss = model(**batch).loss
                print(f"  loss = {loss.item()}  (finite={torch.isfinite(loss).item()})")
                loss.backward()
        else:
            loss = model(**batch).loss
            print(f"  loss = {loss.item()}  (finite={torch.isfinite(loss).item()})")
            loss.backward()
        grad_report(model)


# Track which modules we've already reported, so the FIRST offender per pass is
# obvious rather than a flood of downstream propagation.
_seen: set = set()


def _finite_bad(t) -> str | None:
    if not torch.is_tensor(t) or not t.is_floating_point():
        return None
    if torch.isnan(t).any():
        return "NaN"
    if torch.isinf(t).any():
        return "Inf"
    return None


def install_nan_hooks(model) -> None:
    """Report the first module whose FORWARD output or BACKWARD gradient goes
    non-finite. Forward tells us if the corruption originates in the forward
    pass at all; backward (which fires in reverse order) tells us the deepest
    module still finite, i.e. the NaN is created just downstream of it.
    """
    def fwd_hook(name):
        def hook(module, inputs, output):
            outs = output if isinstance(output, tuple) else (output,)
            for o in outs:
                bad = _finite_bad(o)
                if bad and name not in _seen:
                    _seen.add(name)
                    print(f"    [forward] {bad} first seen leaving: {name} ({type(module).__name__})")
        return hook

    def bwd_hook(name):
        def hook(module, grad_input, grad_output):
            for g in grad_output:
                bad = _finite_bad(g)
                if bad and ("bwd:" + name) not in _seen:
                    _seen.add("bwd:" + name)
                    print(f"    [backward] {bad} in grad arriving at: {name} ({type(module).__name__})")
        return hook

    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(fwd_hook(name))
            module.register_full_backward_hook(bwd_hook(name))


if __name__ == "__main__":
    main()

"""Asymmetric clipping calibration (BitDistiller, arXiv:2402.10631, Eq. 3).

Before QAT starts, searches per-layer clip bounds (alpha, beta) that minimize
the *output* reconstruction error on real calibration activations:

    alpha*, beta* = argmin_{alpha,beta} || Q(Clip(w, alpha, beta)) @ X.T - w @ X.T ||
    w_c = Clip(w, alpha, beta),  alpha in [min_val, 0), beta in (0, max_val]

then permanently clamps the raw pretrained weight to those bounds (Algorithm
1's `w^1 = Clip(w)`, applied once, prior to the training loop). Standard
per-group affine quantization (qat.fake_quant.affine_fake_quant) then proceeds
on the now-clipped tensor for the rest of training -- this only changes the
one-time starting point, not the ongoing per-step quantization.

Must run BEFORE apply_qat() wraps the model's nn.Linear layers, since it
needs the plain nn.Linear modules to hook their real inputs.
"""
import torch
import torch.nn as nn

from .apply_qat import DEFAULT_SKIP_PATTERNS, _should_skip
from .fake_quant import affine_fake_quant

_MAX_CALIB_SAMPLES = 256
_RATIO_STEPS = 10
_MIN_RATIO = 0.5


def _target_linears(model: nn.Module, extra_skip_patterns=()) -> list[tuple[str, nn.Linear]]:
    all_skip = tuple(DEFAULT_SKIP_PATTERNS) + tuple(extra_skip_patterns)
    targets = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not _should_skip(full_name, all_skip):
                targets.append((full_name, child))
    return targets


def _search_clip(weight: torch.Tensor, X: torch.Tensor, bits: int, group_size: int) -> tuple[float, float]:
    """Joint grid search over (alpha, beta), minimizing reconstruction error
    against real calibration activations X ([n_samples, in_features]).
    """
    w_min = weight.min().item()
    w_max = weight.max().item()
    ratios = [1.0 - i * (1.0 - _MIN_RATIO) / (_RATIO_STEPS - 1) for i in range(_RATIO_STEPS)]
    target = weight @ X.T

    best_err = float("inf")
    best_alpha, best_beta = w_min, w_max
    for ra in ratios:
        alpha = w_min * ra
        for rb in ratios:
            beta = w_max * rb
            w_clip = weight.clamp(alpha, beta)
            w_q = affine_fake_quant(w_clip, bits=bits, group_size=group_size)
            err = (w_q @ X.T - target).pow(2).sum().item()
            if err < best_err:
                best_err = err
                best_alpha, best_beta = alpha, beta
    return best_alpha, best_beta


@torch.no_grad()
def calibrate_asymmetric_clipping(
    model: nn.Module,
    calib_batches,
    bits: int,
    group_size: int,
    device,
    extra_skip_patterns=(),
) -> dict:
    """Calibrates and permanently clamps each eligible nn.Linear's weight in
    place, using cached input activations from `calib_batches` (an iterable
    of collated {input_ids, attention_mask, ...} dicts -- labels not needed).

    Returns {layer_name: (alpha, beta)} for logging/diagnostics.
    """
    targets = _target_linears(model, extra_skip_patterns)
    captured: dict[str, list[torch.Tensor]] = {name: [] for name, _ in targets}
    counts: dict[str, int] = {name: 0 for name, _ in targets}

    def make_hook(name):
        def hook(module, inputs):
            if counts[name] >= _MAX_CALIB_SAMPLES:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            captured[name].append(x)
            counts[name] += x.shape[0]
        return hook

    handles = [child.register_forward_pre_hook(make_hook(name)) for name, child in targets]

    was_training = model.training
    model.eval()
    for batch in calib_batches:
        model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    if was_training:
        model.train()

    for h in handles:
        h.remove()

    results = {}
    for name, child in targets:
        if not captured[name]:
            continue
        X = torch.cat(captured[name], dim=0)[:_MAX_CALIB_SAMPLES].float()
        alpha, beta = _search_clip(child.weight.data.float(), X, bits, group_size)
        child.weight.data.copy_(child.weight.data.clamp(alpha, beta))
        results[name] = (alpha, beta)

    return results

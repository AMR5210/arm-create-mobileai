"""Affine (min/max, per-group) fake quantization with a straight-through
estimator (STE), used to simulate low-bit weight quantization during training.

Scale and zero-point are derived from each group's own min/max under
torch.no_grad() (detached from the autograd graph, following standard QAT
practice -- e.g. LSQ/DoReFa-style implementations do the same), so gradients
only flow through the round-to-nearest step, via the well-known
`x + (round(x) - x).detach()` straight-through trick.

Uses the full asymmetric (affine) integer range for a given bit-width
(2**bits distinct codes, via a zero-point), rather than a symmetric range
that would waste one code -- this matters disproportionately at 2 bits,
where every representable level counts.
"""
import torch
import torch.nn.functional as F

_EPS = 1e-8


def affine_fake_quant(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Fake-quantizes a 2D [out_features, in_features] weight tensor.

    Groups are taken along the input-feature dimension. Returns a dequantized
    tensor of the same shape and dtype, differentiable w.r.t. `weight` via STE.
    """
    assert weight.dim() == 2, "affine_fake_quant expects a 2D [out_features, in_features] tensor"
    orig_dtype = weight.dtype
    out_features, in_features = weight.shape

    w = weight.float()
    pad = (-in_features) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    num_groups = w.shape[1] // group_size
    grouped = w.view(out_features, num_groups, group_size)

    with torch.no_grad():
        w_min = grouped.min(dim=-1, keepdim=True).values
        w_max = grouped.max(dim=-1, keepdim=True).values
        levels = 2 ** bits - 1
        scale = (w_max - w_min).clamp_min(_EPS) / levels
        zero_point = torch.round(-w_min / scale).clamp(0, levels)

    x = grouped / scale + zero_point
    x_rounded = x + (torch.round(x) - x).detach()  # STE: identity gradient through round()
    q = torch.clamp(x_rounded, 0, levels)
    dequant = (q - zero_point) * scale

    dequant = dequant.view(out_features, -1)
    if pad:
        dequant = dequant[:, :in_features]
    return dequant.to(orig_dtype)


@torch.no_grad()
def round_trip_init(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """One-shot PTQ-style rounding used only to *initialize* QAT training
    (the plan's progressive FP16 -> INT4 -> INT2 route: initialize the
    trainable shadow weights from an INT4 rounding before fine-tuning them
    toward a 2-bit target), not part of the training graph.
    """
    return affine_fake_quant(weight, bits=bits, group_size=group_size).detach().clone()

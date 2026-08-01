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

    # Min-shift formulation (no explicit integer zero-point). Normalizing by
    # (w - w_min) puts every value in [0, levels] by construction -- so no
    # zero-point is needed, and there is no subtraction of two large numbers.
    #
    # The earlier zero-point form (x = w/scale + round(-w_min/scale)) is
    # numerically fragile precisely where this project fails: for a
    # near-degenerate group (tiny range, common in pretrained weights) scale is
    # small, so the zero-point is huge, and x = w/scale + zero_point subtracts
    # two large near-equal numbers -- catastrophic fp32 cancellation. That gets
    # STRICTLY WORSE at higher bit-widths (more levels -> smaller scale ->
    # larger zero-point), which matched the observed "bits=4 produced NaN
    # gradients where bits=2 did not" report. This form removes that path
    # entirely while producing an equivalent quantization (identity STE
    # Jacobian preserved; outputs differ from the old form by at most one level
    # at the grid boundary, i.e. within quantization noise).
    x = (grouped - w_min) / scale
    x_rounded = x + (torch.round(x) - x).detach()  # STE: identity gradient through round()
    q = torch.clamp(x_rounded, 0, levels)
    dequant = q * scale + w_min

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


# ---------------------------------------------------------------------------
# ParetoQ SEQ (Stretched Elastic Quant), arXiv:2502.02631 Sec 3.2.2.
#
# An ALTERNATIVE to the affine min/max scheme above, opt-in via
# --quant-function seq. The paper reports SEQ beats LSQ-style quantization at
# 2-bit / 1.58-bit (and underperforms it at 3-4 bit). Key differences from the
# affine scheme:
#   * symmetric range [-alpha, alpha] with no zero-point (affine is asymmetric
#     with an implicit zero-point);
#   * alpha is a LEARNABLE per-group scale (an nn.Parameter trained jointly with
#     the weights), initialized to each group's max(|W|) -- not recomputed from
#     min/max stats every forward pass;
#   * the full-precision span is stretched evenly across the levels rather than
#     fitted to the local min/max.
#
# The paper's reference implementation clips to [-1+eps, 1-eps], not a hard
# [-1, 1]: with a hard clip, the group's max weight (|W/alpha| == 1) rounds up
# to one level ABOVE the top code (an overshoot past the intended range). The
# eps keeps the max weight on the top level. eps = 1e-2 matches ParetoQ's
# clip_val = 0.99. Set SEQ_CLIP_EPS = 0 for the literal [-1, 1] form.
SEQ_CLIP_EPS = 1e-2


class _SEQuant(torch.autograd.Function):
    """SEQ forward + the paper's explicit STE gradients.

    Forward (per element, k = 2**bits):
        u   = clip(W_R / alpha, -(1-eps), 1-eps)
        W_Q = alpha * (round(u * k/2 - 0.5) + 0.5) / k * 2

    Backward (STE), with indicator = 1[|W_R/alpha| < 1]:
        dL/dW_R    = grad_out * indicator                       (identity in-range, 0 saturated)
        dL/dalpha  = sum_group( grad_out * (W_Q/alpha - W_R/alpha) * indicator )
    """

    @staticmethod
    def forward(ctx, w, alpha, k, clip_eps):
        u = w / alpha
        clip_hi = 1.0 - clip_eps
        u_c = torch.clamp(u, -clip_hi, clip_hi)
        q = torch.round(u_c * (k / 2.0) - 0.5) + 0.5
        w_q = alpha * q * (2.0 / k)
        ctx.save_for_backward(u, w_q, alpha)
        return w_q

    @staticmethod
    def backward(ctx, grad_out):
        u, w_q, alpha = ctx.saved_tensors
        indicator = (u.abs() < 1.0).to(grad_out.dtype)
        grad_w = grad_out * indicator
        # dL/dalpha (per element) = grad_out * (W_Q/alpha - W_R/alpha) * indicator;
        # alpha is shared within a group, so sum over the group (last) axis.
        grad_alpha_elem = grad_out * (w_q / alpha - u) * indicator
        grad_alpha = grad_alpha_elem.sum(dim=-1, keepdim=True)
        return grad_w, grad_alpha, None, None


def seq_alpha_init(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Initial per-group SEQ scale: max(|W|) within each group of `group_size`
    consecutive input features. Returns shape [out_features, num_groups].
    """
    assert weight.dim() == 2
    out_features, in_features = weight.shape
    w = weight.detach().float()
    pad = (-in_features) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    num_groups = w.shape[1] // group_size
    grouped = w.view(out_features, num_groups, group_size)
    # clamp_min avoids a zero scale (division by zero) for an all-zero group.
    return grouped.abs().amax(dim=-1).clamp_min(_EPS)


def seq_fake_quant(
    weight: torch.Tensor, alpha: torch.Tensor, bits: int, group_size: int,
    clip_eps: float = SEQ_CLIP_EPS,
) -> torch.Tensor:
    """SEQ fake-quantization of a 2D [out, in] weight with a learnable per-group
    `alpha` (shape [out, num_groups]). Differentiable w.r.t. BOTH `weight` and
    `alpha` via the STE rules in _SEQuant.
    """
    assert weight.dim() == 2, "seq_fake_quant expects a 2D [out_features, in_features] tensor"
    orig_dtype = weight.dtype
    out_features, in_features = weight.shape

    w = weight.float()
    pad = (-in_features) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    num_groups = w.shape[1] // group_size
    grouped = w.view(out_features, num_groups, group_size)
    # alpha is a free nn.Parameter, so nothing stops the optimizer from driving
    # it to zero or negative -- but SEQ defines the symmetric range
    # [-alpha, alpha], which only means anything for alpha > 0. Two failure
    # modes, both observed on a real 1000-step run (3 of 26.3M scales went
    # negative, min -3.0e-05):
    #   * alpha < 0 silently MIRRORS that group's quantization grid;
    #   * alpha == 0 makes the backward's W_Q/alpha a 0/0 NaN, which the
    #     |u| < 1 indicator cannot mask (nan * 0 == nan), poisoning AdamW's
    #     moment buffers for good.
    # Take the magnitude and floor it. abs() (rather than clamp alone) keeps a
    # group that crossed zero trainable via its magnitude instead of freezing
    # it at the floor, where clamp's gradient is zero. This is an identity for
    # every alpha > _EPS, so healthy groups -- i.e. essentially all of them --
    # are bit-identical to before.
    a = alpha.float().abs().clamp_min(_EPS).view(out_features, num_groups, 1)

    k = float(2 ** bits)
    dequant = _SEQuant.apply(grouped, a, k, clip_eps)

    dequant = dequant.view(out_features, -1)
    if pad:
        dequant = dequant[:, :in_features]
    return dequant.to(orig_dtype)

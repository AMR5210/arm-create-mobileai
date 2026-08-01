"""Distillation objectives for QAT (teacher = frozen fp16, student = quantized).

Isolated from the quantization math: these only consume logits + a response
mask and return a scalar loss. CAKLD lives in train_qat.py; this module adds the
two alternatives selectable via --distill-loss:

  * kl          -- standard forward KL(teacher || student), response tokens only.
  * wasserstein -- sliced-Wasserstein distribution-alignment loss
                   (arXiv:2601.07878, Jan 2026), which that paper reports beats
                   KL-based distillation for ultra-low-bit LLM quantization.

Sliced-Wasserstein idea: KL matches teacher/student PER TOKEN. Instead, treat
the batch's response-token outputs as two empirical distributions of vectors in
R^V (V = vocab), and align those DISTRIBUTIONS. The Wasserstein distance between
two high-dimensional distributions is intractable, but the SLICED approximation
is cheap: project both onto many random 1D directions, and on each 1D line the
Wasserstein-p distance between two equal-size sample sets has a closed form --
sort both, take the L_p distance between the sorted values (optimal transport in
1D is monotone). Average over directions.

NOTE (defaults to confirm against the paper -- arXiv:2601.07878 could not be
fetched here, only its abstract): this implements the standard SW estimator,
which is unambiguous, but four representation choices are the paper's to pin
down and are exposed as parameters with documented defaults:
  - REPRESENTATION: which "output distribution" is projected -- softmax
    probabilities (default, bounded/stable), log-probabilities, or raw logits.
  - p: Wasserstein order. Default 1 (mean absolute sorted difference).
  - num_projections L: default 64.
  - directions sampled i.i.d. Gaussian, normalized to the unit sphere.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_NUM_PROJECTIONS = 64
DEFAULT_REPRESENTATION = "prob"   # "prob" | "logprob" | "logit"


def _represent(logits: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "prob":
        return logits.softmax(dim=-1)
    if representation == "logprob":
        return logits.log_softmax(dim=-1)
    if representation == "logit":
        return logits
    raise ValueError(f"unknown representation {representation!r}")


def sliced_wasserstein(
    x: torch.Tensor, y: torch.Tensor, num_projections: int = DEFAULT_NUM_PROJECTIONS,
    p: int = 1, directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sliced-Wasserstein-p distance between two equal-size sample sets
    x, y of shape [N, D] (N samples each, D dims). Differentiable w.r.t. x
    (y should be detached by the caller for a teacher target).

    directions: optional [L, D] projection matrix (unit rows) to inject for
    reproducible tests; sampled fresh each call when None.
    """
    assert x.shape == y.shape and x.dim() == 2, (x.shape, y.shape)
    n, d = x.shape
    if directions is None:
        directions = torch.randn(num_projections, d, device=x.device, dtype=x.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # project: [N, D] @ [D, L] -> [N, L]
    px = x @ directions.t()
    py = y @ directions.t()
    # 1D Wasserstein-p on each column: sort both, L_p between sorted values.
    sx = px.sort(dim=0).values
    sy = py.sort(dim=0).values
    diff = (sx - sy).abs()
    if p != 1:
        diff = diff.pow(p)
    return diff.mean()  # mean over samples and projections


def _response_rows(student_logits, teacher_logits, labels):
    """Shift + response-token mask (same convention as cakld_loss), returning the
    masked student/teacher logit rows [N, V] (or None if no response tokens)."""
    shift_student = student_logits[..., :-1, :].float()
    shift_teacher = teacher_logits[..., :-1, :].float()
    mask = labels[..., 1:] != -100
    if not mask.any():
        return None, None
    return shift_student[mask], shift_teacher[mask].detach()


def sliced_wasserstein_distill_loss(
    student_logits, teacher_logits, labels,
    num_projections: int = DEFAULT_NUM_PROJECTIONS,
    representation: str = DEFAULT_REPRESENTATION, p: int = 1,
) -> torch.Tensor:
    """Sliced-Wasserstein distribution-alignment distillation loss over the
    batch's response tokens (arXiv:2601.07878)."""
    student_rows, teacher_rows = _response_rows(student_logits, teacher_logits, labels)
    if student_rows is None:
        return student_logits.new_zeros(())
    s = _represent(student_rows, representation)
    t = _represent(teacher_rows, representation)
    return sliced_wasserstein(s, t, num_projections=num_projections, p=p)


def kl_distill_loss(student_logits, teacher_logits, labels) -> torch.Tensor:
    """Standard forward KL(teacher || student), response tokens only -- the
    classic distillation direction, as a baseline alternative to CAKLD."""
    student_rows, teacher_rows = _response_rows(student_logits, teacher_logits, labels)
    if student_rows is None:
        return student_logits.new_zeros(())
    student_lp = student_rows.log_softmax(dim=-1)
    teacher_lp = teacher_rows.log_softmax(dim=-1)
    # KL(T || S) = sum_c p_T (log p_T - log p_S), mean over tokens.
    kl = (teacher_lp.exp() * (teacher_lp - student_lp)).sum(dim=-1)
    return kl.mean()

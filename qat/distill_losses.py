"""Distillation objectives for QAT (teacher = frozen fp16, student = quantized).

Isolated from the quantization math. CAKLD lives in train_qat.py; this module
adds the two alternatives selectable via --distill-loss:

  * kl          -- standard forward KL(teacher || student) over final-logit
                   response tokens.
  * wasserstein -- the sliced-Wasserstein + MSE HIDDEN-STATE distribution
                   alignment of arXiv:2601.07878 (Jan 2026), which that paper
                   reports beats KL-based distillation for ultra-low-bit LLM
                   quantization.

Where the loss is applied (arXiv:2601.07878, Eq 1-5): NOT on final logits, but
on the real-valued hidden states at selected transformer BLOCK boundaries. For a
block, flatten the teacher (fp16) and student (quantized) hidden states
[batch, seq, hidden] -> [batch*seq, hidden] and align those two empirical
distributions of activation vectors with

    L_block = (1 - sw_w) * MSE(Y_fp, Y_q) + sw_w * SW(Y_fp, Y_q)

MSE pulls each position's student activation toward the teacher's (pointwise);
SW additionally matches the two activation DISTRIBUTIONS as a whole.

Sliced-Wasserstein (the SW term): the Wasserstein distance between two
high-dimensional distributions is intractable, but the SLICED approximation is
cheap -- project both onto many random 1D directions, and on each line the
Wasserstein-p distance between two equal-size sample sets has a closed form:
sort both, take the L_p distance between the sorted values (optimal transport in
1D is monotone). Average over directions. p = 1 (mean absolute) per the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NUM_PROJECTIONS = 256   # paper sweeps 128-1024 with diminishing returns; 256 is a cost/quality middle
DEFAULT_SW_WEIGHT = 0.1         # paper Fig 5 sweet spot is intermediate (~0.05-0.2), not high


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


def block_sw_mse_loss(
    student_rows: torch.Tensor, teacher_rows: torch.Tensor,
    sw_weight: float = DEFAULT_SW_WEIGHT, num_projections: int = DEFAULT_NUM_PROJECTIONS,
    p: int = 1,
) -> torch.Tensor:
    """arXiv:2601.07878 per-block loss on hidden states:
        (1 - sw_w) * MSE + sw_w * SW,
    between teacher (fp16) and student (quantized) block hidden-state rows, each
    already flattened+masked to [N, hidden]. Differentiable w.r.t. the student;
    the teacher target is detached.
    """
    s = student_rows.float()
    t = teacher_rows.float().detach()
    assert s.shape == t.shape, (s.shape, t.shape)
    if s.shape[0] == 0:                       # no positions after masking
        return s.new_zeros(())
    mse = F.mse_loss(s, t)
    sw = sliced_wasserstein(s, t, num_projections=num_projections, p=p)
    return (1.0 - sw_weight) * mse + sw_weight * sw


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The transformer decoder-block ModuleList (Qwen3/Llama: model.model.layers)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError(
        "could not locate decoder layers (expected model.model.layers); "
        "adjust _decoder_layers for this architecture."
    )


class BlockHiddenStateHooks:
    """Forward hooks that capture the hidden-state output of selected decoder
    blocks during a normal forward pass -- so no EXTRA forward is needed (the
    student's main forward and the teacher's existing distillation forward both
    already run). Read `.captured[i]` after the forward; overwritten each pass.
    """

    def __init__(self, model: nn.Module, layer_indices):
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        layers = _decoder_layers(model)
        n = len(layers)
        for i in layer_indices:
            if not (0 <= i < n):
                self.remove()
                raise IndexError(f"block index {i} out of range for a {n}-layer model")
            self.handles.append(layers[i].register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def hook(module, inputs, output):
            # decoder layers return either a tensor or a tuple whose [0] is the
            # hidden state.
            self.captured[i] = output[0] if isinstance(output, tuple) else output
        return hook

    def clear(self):
        self.captured.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def combined_block_loss(
    student_caps: dict, teacher_caps: dict, layer_indices, mask: torch.Tensor | None = None,
    sw_weight: float = DEFAULT_SW_WEIGHT, num_projections: int = DEFAULT_NUM_PROJECTIONS,
    p: int = 1,
) -> torch.Tensor:
    """Mean over the selected blocks of block_sw_mse_loss. Averaging (not
    summing) keeps the loss scale independent of how many blocks are selected.

    mask: optional [batch, seq] boolean selecting which positions to compare
    (e.g. labels != -100 -- the same response-only / non-padding mask the hard
    loss and CAKLD use). Padded (and, for instruction data, prompt) positions
    are excluded so variable-length examples don't add noise. When None, all
    positions are used (flatten [B,S,H] -> [B*S,H])."""
    losses = []
    for i in layer_indices:
        sh, th = student_caps[i], teacher_caps[i]           # [B, S, H]
        if mask is not None:
            s_rows, t_rows = sh[mask], th[mask]             # [N, H]
        else:
            s_rows = sh.reshape(-1, sh.shape[-1])
            t_rows = th.reshape(-1, th.shape[-1])
        losses.append(block_sw_mse_loss(s_rows, t_rows, sw_weight, num_projections, p))
    return torch.stack(losses).mean()


def _response_rows(student_logits, teacher_logits, labels):
    """Shift + response-token mask (same convention as cakld_loss), returning the
    masked student/teacher logit rows [N, V] (or None if no response tokens)."""
    shift_student = student_logits[..., :-1, :].float()
    shift_teacher = teacher_logits[..., :-1, :].float()
    mask = labels[..., 1:] != -100
    if not mask.any():
        return None, None
    return shift_student[mask], shift_teacher[mask].detach()


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

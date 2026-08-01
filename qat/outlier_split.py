"""Outlier Channel Splitting (OCS) for QAT -- an alternative to the
full-precision --skip-layers escape hatch for outlier-heavy layers.

Idea (Zhao et al., 2019, "Improving Neural Network Quantization without
Retraining using Outlier Channel Splitting"): an input channel carrying an
outlier weight inflates its quantization group's range, coarsening every other
weight in that group. Split that channel into two channels each carrying half
the magnitude, and duplicate the corresponding input feature so both halves see
the same activation. Because
        x[c]*(W[:,c]/2) + x[c]*(W[:,c]/2) == x[c]*W[:,c],
the split is an EXACT identity on the forward pass, but each half now has a
smaller magnitude to quantize -- so an outlier-heavy channel becomes quantizable
at full 2-bit instead of needing a full-precision bypass.

Deployment invisibility (the hard constraint): llama.cpp / GGUF have no concept
of channel splitting. So the split lives ONLY during training. Before export we
MERGE each split pair back into one channel (sum the two columns), restoring the
exact original [out_features, in_features] shape -- the exported GGUF is a
completely standard, unsplit model. The merge
        W_merged[:, c] = W_split_a[:, c] + W_split_b[:, c]
is an exact fold for ANY weights (proof: the split matmul equals
x @ W_merged^T identically, at every point in training, not just at init), so
nothing about the tensor layout leaks to the exporter.

Design for isolation / easy revert: this is a self-contained module. Splitting
wraps a target nn.Linear in an OutlierSplitLinear whose INNER linear has the
expanded input dimension; apply_qat then wraps that inner linear with
FakeQuantLinear exactly as it would any nn.Linear (so the fake-quant operates on
the reduced-magnitude split columns). Merging replaces the wrapper with a plain
nn.Linear again. Nothing in fake_quant.py / apply_qat.py / the exporter changes;
when the feature is off, none of this code runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_outlier_channels(weight: torch.Tensor, n: int) -> list[int]:
    """Indices of the `n` input channels (columns of a [out, in] weight) with
    the largest peak magnitude -- the ones whose outlier most inflates their
    quantization group. Ties broken by index for determinism.
    """
    assert weight.dim() == 2
    col_peak = weight.detach().abs().amax(dim=0)  # [in]
    n = max(0, min(n, weight.shape[1]))
    if n == 0:
        return []
    return sorted(torch.topk(col_peak, n).indices.tolist())


class OutlierSplitLinear(nn.Module):
    """Wraps a Linear whose input channels have been expanded by splitting the
    selected outlier columns into half-magnitude pairs. The forward augments the
    input with copies of those channels so the expanded matmul reproduces the
    original layer's output exactly.

    Layout: the expanded inner linear keeps the original `orig_in` columns
    (with each split column halved IN PLACE) and appends one extra column per
    split (the other half). `split_sources[m]` is the original column index whose
    second half lives at column `orig_in + m`.
    """

    def __init__(self, inner: nn.Linear, split_sources: list[int], orig_in: int):
        super().__init__()
        self.linear = inner                 # nn.Linear [out, orig_in + len(split_sources)]
        self.split_sources = list(split_sources)
        self.orig_in = orig_in

    @classmethod
    def from_linear(cls, linear: nn.Linear, channels: list[int]) -> "OutlierSplitLinear":
        out_features, orig_in = linear.weight.shape
        channels = list(channels)
        k = len(channels)
        expanded = nn.Linear(orig_in + k, out_features, bias=linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.data.clone()          # [out, orig_in]
            halves = w[:, channels] * 0.5           # [out, k]  second halves
            w[:, channels] = w[:, channels] * 0.5   # halve in place (first halves)
            expanded.weight.data.copy_(torch.cat([w, halves], dim=1))
            if linear.bias is not None:
                expanded.bias.data.copy_(linear.bias.data)
        return cls(expanded, channels, orig_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Duplicate the split-source input features into the appended columns so
        # both halves of each split channel receive the same activation.
        if self.split_sources:
            extra = x[..., self.split_sources]
            x = torch.cat([x, extra], dim=-1)
        return self.linear(x)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the appended half-columns back into their originals, yielding a
        standard nn.Linear with the ORIGINAL input dimension. Exact for any
        current weights -- call after materialize (so self.linear holds the
        baked, quantized weight) to produce the checkpoint that gets exported.
        """
        w = self.linear.weight.data.clone()         # [out, orig_in + k]
        base = w[:, : self.orig_in].clone()
        for m, c in enumerate(self.split_sources):
            base[:, c] = base[:, c] + w[:, self.orig_in + m]
        merged = nn.Linear(self.orig_in, w.shape[0], bias=self.linear.bias is not None)
        merged.weight.data.copy_(base)
        if self.linear.bias is not None:
            merged.bias.data.copy_(self.linear.bias.data)
        return merged


def _matches(name: str, patterns) -> bool:
    return any(p in name for p in patterns)


def apply_outlier_split(model: nn.Module, patterns, n_channels: int = 1) -> list[dict]:
    """Replace each nn.Linear whose name matches `patterns` with an
    OutlierSplitLinear that splits its `n_channels` worst outlier input channels.
    Run BEFORE apply_qat so the (expanded) inner linears get fake-quant-wrapped.
    Returns per-layer split info. A no-op when `patterns` is empty.
    """
    if not patterns:
        return []
    info = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _matches(full, patterns):
                channels = select_outlier_channels(child.weight, n_channels)
                if not channels:
                    continue
                setattr(parent, child_name, OutlierSplitLinear.from_linear(child, channels))
                info.append({"layer": full, "channels": channels, "orig_in": child.weight.shape[1]})
    return info


def merge_outlier_split(model: nn.Module) -> list[str]:
    """Replace every OutlierSplitLinear with its merged plain nn.Linear (original
    shape restored). Run AFTER materialize_qat, BEFORE save_pretrained/export.
    Returns the merged layer names. A no-op when nothing was split.
    """
    merged = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, OutlierSplitLinear):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                setattr(parent, child_name, child.merge())
                merged.append(full)
    return merged

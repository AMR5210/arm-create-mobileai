"""Picks a safe `attn_implementation` for `AutoModelForCausalLM.from_pretrained`.

On ROCm, PyTorch's SDPA dispatcher defaults to the "memory efficient"
attention backend (flash attention refuses this model's GQA + additive
attn_mask shape and raises, so mem-efficient is what's actually selected).
That backend's backward pass produces NaN gradients on MI300X (ROCm 6.2,
torch 2.5.1+rocm6.2) for every parameter upstream of the corrupted layer --
confirmed by diagnostics that show forward activations all finite, and
gradients finite down through some middle layer, then NaN in every earlier
layer once the fused backward kernel is hit. Forcing "eager" attention (the
plain, unfused reference implementation) makes the same run produce zero
NaN/Inf gradients. The same code path is fine on CUDA, where SDPA dispatches
to a different backend, so this only needs to override on ROCm.
"""
import torch


def safe_attn_implementation() -> str:
    if torch.version.hip is not None:
        return "eager"
    return "sdpa"

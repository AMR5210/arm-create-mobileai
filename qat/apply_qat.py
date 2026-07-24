"""Walks a Hugging Face model and swaps target nn.Linear layers in-place for
FakeQuantLinear (QAT training), or back to plain nn.Linear (export).

Embeddings, the LM head, and norm layers are left at full precision by
default -- standard practice in weight-only LLM quantization (and consistent
with llama.cpp's own Q2_K "K-quant" scheme, which keeps certain tensors at
higher precision within a nominally-2-bit model).
"""
import torch
import torch.nn as nn

from .fake_quant import affine_fake_quant
from .quantized_linear import FakeQuantLinear

DEFAULT_SKIP_PATTERNS = ("embed_tokens", "lm_head", "norm")


def _should_skip(name: str, skip_patterns) -> bool:
    return any(pattern in name for pattern in skip_patterns)


def apply_qat(
    model: nn.Module,
    bits: int = 2,
    group_size: int = 32,
    init_bits: int = 4,
    skip_patterns=DEFAULT_SKIP_PATTERNS,
) -> list[str]:
    """Replaces eligible nn.Linear submodules with FakeQuantLinear, in place.
    Returns the list of replaced module names.
    """
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not _should_skip(full_name, skip_patterns):
                qlinear = FakeQuantLinear.from_linear(
                    child, bits=bits, group_size=group_size, init_bits=init_bits
                )
                setattr(parent, child_name, qlinear)
                replaced.append(full_name)
    return replaced


def materialize_qat(model: nn.Module) -> list[str]:
    """Bakes every FakeQuantLinear back into a plain nn.Linear, in place.
    Call this before save_pretrained() so the checkpoint needs no custom
    module classes to load.
    """
    materialized = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, FakeQuantLinear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                setattr(parent, child_name, child.materialize())
                materialized.append(full_name)
    return materialized


def fake_quantized_state_dict(model: nn.Module) -> dict:
    """Returns a state dict with every FakeQuantLinear's *current*
    fake-quantized weight baked in, WITHOUT mutating the live model (unlike
    materialize_qat, which replaces modules in place and would silently break
    training if you tried to keep going afterward).

    Meant for periodic crash/disconnect-safe checkpointing during a long
    training run (e.g. on a free Colab session, where disconnects are
    common) -- safe to call mid-loop, training continues unaffected. This
    only preserves weights, not optimizer state, so resuming from one of
    these is a fallback ("train from this partial point" or "just export
    what we have"), not a true training resume.
    """
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinear):
            with torch.no_grad():
                w_hat = affine_fake_quant(module.weight, module.bits, module.group_size)
            state_dict[f"{name}.weight"] = w_hat.detach().cpu().clone()
            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.detach().cpu().clone()

    for key, tensor in model.state_dict().items():
        if key not in state_dict:
            state_dict[key] = tensor.detach().cpu().clone()

    return state_dict

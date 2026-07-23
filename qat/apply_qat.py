"""Walks a Hugging Face model and swaps target nn.Linear layers in-place for
FakeQuantLinear (QAT training), or back to plain nn.Linear (export).

Embeddings, the LM head, and norm layers are left at full precision by
default -- standard practice in weight-only LLM quantization (and consistent
with llama.cpp's own Q2_K "K-quant" scheme, which keeps certain tensors at
higher precision within a nominally-2-bit model).
"""
import torch.nn as nn

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

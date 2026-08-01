"""A drop-in replacement for nn.Linear that fake-quantizes its weight on every
forward pass, and can bake the current fake-quantized weight back into a plain
nn.Linear for saving/export once training is done.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import affine_fake_quant, round_trip_init, seq_alpha_init, seq_fake_quant


def ptq_q2k_dequant_init(weight: torch.Tensor) -> torch.Tensor | None:
    """Dequantized GGUF Q2_K of the weight -- i.e. the actual PTQ-2bit weights,
    for use as the QAT shadow-weight initialization instead of a naive round-trip.

    Runs the same Q2_K quantiser the export/PTQ-baseline path uses
    (qat/gguf_q2k) on the raw fp16 weight and decodes it back to fp32, so the
    QAT shadow weights start exactly where the deployed PTQ checkpoint sits.
    Research on low-bit QAT (e.g. BitDistiller) shows this converges better than
    initializing from a plain fake-quant round-trip of the fp16 weights.

    Returns None when the weight can't be Q2_K-encoded (in_features not a
    multiple of Q2_K's 256-element super-block, or not 2D), so the caller falls
    back to the round-trip init for that layer.
    """
    import numpy as np

    from .gguf_q2k import QK_K, dequantize_q2_k, quantize_q2_k

    w = weight.detach().float().cpu().numpy()
    if w.ndim != 2 or w.shape[1] % QK_K != 0:
        return None
    packed = quantize_q2_k(np.ascontiguousarray(w))
    deq = dequantize_q2_k(packed, w.shape[1])
    return torch.from_numpy(np.ascontiguousarray(deq)).to(dtype=weight.dtype, device=weight.device)


class FakeQuantLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, group_size: int,
                 quant_function: str = "affine"):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.bits = bits
        self.group_size = group_size
        self.quant_function = quant_function
        # SEQ (ParetoQ) uses a learnable per-group scale trained jointly with the
        # weights; register it as a Parameter so it lands in model.parameters()
        # and the optimizer picks it up exactly like weight/bias. affine has no
        # such parameter (its scale is recomputed from group min/max each pass).
        if quant_function == "seq":
            self.alpha = nn.Parameter(seq_alpha_init(self.weight.data, group_size))
        else:
            self.alpha = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 2,
        group_size: int = 16,  # Q2_K sub-block aligned (see scripts/export_qat_gguf.py)
        init_bits: int = 4,
        init_mode: str = "roundtrip",
        quant_function: str = "affine",
    ) -> "FakeQuantLinear":
        """init_mode:
          "roundtrip" -- shadow weights = INT4 fake-quant round-trip of fp16 (default).
          "ptq_q2k"   -- shadow weights = dequantized Q2_K (the real PTQ-2bit
                         weights); falls back to "roundtrip" for layers Q2_K
                         can't encode (in_features not a multiple of 256).
        quant_function -- "affine" (min/max, default) or "seq" (ParetoQ SEQ).
        """
        init_weight = None
        used_ptq = False
        if init_mode == "ptq_q2k":
            init_weight = ptq_q2k_dequant_init(linear.weight.data)
            used_ptq = init_weight is not None
        if init_weight is None:
            init_weight = round_trip_init(linear.weight.data, bits=init_bits, group_size=group_size)
        bias = linear.bias.data if linear.bias is not None else None
        module = cls(init_weight, bias, bits=bits, group_size=group_size,
                     quant_function=quant_function)
        module.init_mode_used = "ptq_q2k" if used_ptq else "roundtrip"
        return module

    def quantized_weight(self) -> torch.Tensor:
        """Current fake-quantized weight per the configured quant_function.
        Differentiable (STE) w.r.t. self.weight (and self.alpha for SEQ)."""
        if self.quant_function == "seq":
            return seq_fake_quant(self.weight, self.alpha, self.bits, self.group_size)
        return affine_fake_quant(self.weight, self.bits, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)

    def materialize(self) -> nn.Linear:
        """Bakes the current fake-quantized weight into a plain nn.Linear,
        so the model can be saved/converted with ordinary HF/GGUF tooling.
        """
        with torch.no_grad():
            w_hat = self.quantized_weight()
        out_features, in_features = w_hat.shape
        linear = nn.Linear(in_features, out_features, bias=self.bias is not None)
        linear.weight.data.copy_(w_hat)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear

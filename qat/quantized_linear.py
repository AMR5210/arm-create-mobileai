"""A drop-in replacement for nn.Linear that fake-quantizes its weight on every
forward pass, and can bake the current fake-quantized weight back into a plain
nn.Linear for saving/export once training is done.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import affine_fake_quant, round_trip_init


class FakeQuantLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, group_size: int):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.bits = bits
        self.group_size = group_size

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 2,
        group_size: int = 32,
        init_bits: int = 4,
    ) -> "FakeQuantLinear":
        init_weight = round_trip_init(linear.weight.data, bits=init_bits, group_size=group_size)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(init_weight, bias, bits=bits, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = affine_fake_quant(self.weight, self.bits, self.group_size)
        return F.linear(x, w_hat, self.bias)

    def materialize(self) -> nn.Linear:
        """Bakes the current fake-quantized weight into a plain nn.Linear,
        so the model can be saved/converted with ordinary HF/GGUF tooling.
        """
        with torch.no_grad():
            w_hat = affine_fake_quant(self.weight, self.bits, self.group_size)
        out_features, in_features = w_hat.shape
        linear = nn.Linear(in_features, out_features, bias=self.bias is not None)
        linear.weight.data.copy_(w_hat)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear

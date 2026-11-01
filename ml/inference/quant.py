"""WS-5 quantization — 8-bit stub for CPU/GPU budget fit."""

from __future__ import annotations

from torch import nn


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """Fake 8-bit quantize for smoke: scale weights to int8 range and back."""
    if bits not in (8, 4):
        raise ValueError(f"unsupported bits {bits}, expected 8 or 4")
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        # Simple per-tensor fake quant
        scale = param.abs().max().clamp(min=1e-6) / (2 ** (bits - 1) - 1)
        q = (param / scale).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
        param.data = (q * scale).to(param.dtype)
    # Tag
    model._quant_bits = bits  # type: ignore
    return model

def dequantize_model(model: nn.Module) -> nn.Module:
    # No-op for stub (weights already dequantized)
    if hasattr(model, "_quant_bits"):
        delattr(model, "_quant_bits")
    return model

"""Inference-only INT8 and packed INT4 weight quantization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _quantize_weight(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_value = weight.detach().float().abs().amax(dim=1, keepdim=True)
    if not torch.isfinite(max_value).all():
        raise ValueError("cannot quantize weights containing NaN or infinity")

    qmax = 2 ** (bits - 1) - 1
    scale = torch.where(max_value == 0, torch.ones_like(max_value), max_value / qmax)
    quantized = (weight.detach().float() / scale).round().clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return quantized, scale

    columns = quantized.shape[1]
    if columns % 2:
        quantized = F.pad(quantized, (0, 1))
    values = quantized.to(torch.int16) & 0xF
    packed = values[:, 0::2] | (values[:, 1::2] << 4)
    return packed.to(torch.uint8), scale


def _dequantize_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, int],
    bits: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bits == 8:
        quantized = qweight
    else:
        values = qweight.to(torch.int16)
        low = values & 0xF
        high = values >> 4
        quantized = torch.stack((low, high), dim=-1).flatten(start_dim=-2)
        quantized = torch.where(quantized >= 8, quantized - 16, quantized)
        quantized = quantized[:, : shape[1]]
    return (quantized.to(torch.float32) * scale).to(dtype=dtype)


class _QuantizedLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bits: int,
        weight_key: int,
    ):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight_shape = tuple(source.weight.shape)
        self.weight_dtype = source.weight.dtype
        self.weight_requires_grad = source.weight.requires_grad
        self.bits = bits
        self.weight_key = weight_key
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_parameter("bias", source.bias)
        self.train(source.training)

    def _weight(self, dtype: torch.dtype) -> torch.Tensor:
        return _dequantize_weight(
            self.qweight, self.scale, self.weight_shape, self.bits, dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self._weight(input.dtype), self.bias)


class _QuantizedEmbedding(nn.Module):
    def __init__(
        self,
        source: nn.Embedding,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bits: int,
        weight_key: int,
    ):
        super().__init__()
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.weight_shape = tuple(source.weight.shape)
        self.weight_dtype = source.weight.dtype
        self.weight_requires_grad = source.weight.requires_grad
        self.bits = bits
        self.weight_key = weight_key
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.train(source.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = _dequantize_weight(
            self.qweight, self.scale, self.weight_shape, self.bits, self.weight_dtype
        )
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


_QuantizedModule = _QuantizedLinear | _QuantizedEmbedding


def _replace_with_quantized(
    module: nn.Module,
    bits: int,
    weight_keys: dict[int, int],
    quantized_weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
    source_weights: list[nn.Parameter],
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Linear, nn.Embedding)):
            if isinstance(child, nn.Embedding) and child.max_norm is not None:
                raise ValueError("quantized embeddings do not support max_norm")
            weight_id = id(child.weight)
            weight_key = weight_keys.get(weight_id)
            if weight_key is None:
                weight_key = len(weight_keys)
                weight_keys[weight_id] = weight_key
                source_weights.append(child.weight)
                quantized_weights[weight_key] = _quantize_weight(child.weight, bits)
            qweight, scale = quantized_weights[weight_key]
            if isinstance(child, nn.Linear):
                replacement = _QuantizedLinear(child, qweight, scale, bits, weight_key)
            else:
                replacement = _QuantizedEmbedding(child, qweight, scale, bits, weight_key)
            setattr(module, name, replacement)
        else:
            _replace_with_quantized(
                child, bits, weight_keys, quantized_weights, source_weights
            )


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """Replace linear and embedding weights with INT8 or packed INT4 storage."""
    if bits not in (8, 4):
        raise ValueError(f"unsupported bits {bits}, expected 8 or 4")
    current_bits = getattr(model, "_quant_bits", None)
    if current_bits == bits:
        return model
    if current_bits in (8, 4):
        model = dequantize_model(model)

    is_layer = isinstance(model, (nn.Linear, nn.Embedding))
    container = nn.Module()
    if is_layer:
        container.add_module("root", model)
    else:
        container = model

    supported_weights = {
        id(module.weight)
        for module in container.modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    }
    unsupported = [
        name
        for name, parameter in container.named_parameters()
        if parameter.dim() >= 2 and id(parameter) not in supported_weights
    ]
    if unsupported:
        raise ValueError(
            f"cannot quantize matrix parameters outside linear/embedding layers: {unsupported}"
        )

    weight_keys: dict[int, int] = {}
    quantized_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    source_weights: list[nn.Parameter] = []
    _replace_with_quantized(
        container, bits, weight_keys, quantized_weights, source_weights
    )
    result = container.root if is_layer else model
    result._quant_bits = bits  # type: ignore[attr-defined]
    return result


def _restore_module(
    source: _QuantizedModule, tied_weights: dict[int, nn.Parameter]
) -> nn.Module:
    weight = tied_weights.get(source.weight_key)
    if weight is None:
        value = _dequantize_weight(
            source.qweight,
            source.scale,
            source.weight_shape,
            source.bits,
            source.weight_dtype,
        )
        weight = nn.Parameter(value, requires_grad=source.weight_requires_grad)
        tied_weights[source.weight_key] = weight

    if isinstance(source, _QuantizedLinear):
        restored = nn.Linear(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.qweight.device,
            dtype=source.weight_dtype,
        )
        restored.weight = weight
        restored.bias = source.bias
    else:
        restored = nn.Embedding(
            source.num_embeddings,
            source.embedding_dim,
            padding_idx=source.padding_idx,
            max_norm=source.max_norm,
            norm_type=source.norm_type,
            scale_grad_by_freq=source.scale_grad_by_freq,
            sparse=source.sparse,
            device=source.qweight.device,
            dtype=source.weight_dtype,
        )
        restored.weight = weight
    restored.train(source.training)
    return restored


def dequantize_model(model: nn.Module) -> nn.Module:
    """Restore quantized layers to floating-point modules."""
    tied_weights: dict[int, nn.Parameter] = {}
    if isinstance(model, (_QuantizedLinear, _QuantizedEmbedding)):
        return _restore_module(model, tied_weights)

    def restore_children(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (_QuantizedLinear, _QuantizedEmbedding)):
                setattr(module, name, _restore_module(child, tied_weights))
            else:
                restore_children(child)

    restore_children(model)
    if hasattr(model, "_quant_bits"):
        delattr(model, "_quant_bits")
    return model

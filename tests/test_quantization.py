import io

import pytest
import torch
from torch import nn

from ml.inference.quant import dequantize_model, quantize_model
from ml.models import KothaGPT
from ml.models.config import ModelConfig


def _small_model():
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=64,
    )
    return KothaGPT(config).eval()


def _storage_bytes(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@pytest.mark.parametrize(("bits", "tolerance"), [(8, 0.02), (4, 0.15)])
def test_quantization_storage_quality_and_state_dict_round_trip(bits, tolerance):
    torch.manual_seed(7)
    model = _small_model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline_logits = model(tokens)["logits"]
    baseline_bytes = _storage_bytes(model)

    quantize_model(model, bits=bits)

    assert model._quant_bits == bits
    assert not any(parameter.dim() >= 2 for parameter in model.parameters())
    assert model.embed_tokens.qweight.dtype == (torch.int8 if bits == 8 else torch.uint8)
    storage_ratio = _storage_bytes(model) / baseline_bytes
    assert storage_ratio < (0.7 if bits == 8 else 0.3)
    with torch.no_grad():
        quantized_logits = model(tokens)["logits"]
    relative_rms_error = (
        torch.mean((baseline_logits - quantized_logits).square()).sqrt()
        / torch.mean(baseline_logits.square()).sqrt()
    )
    assert relative_rms_error.item() < tolerance

    state_bytes = io.BytesIO()
    torch.save(model.state_dict(), state_bytes)
    state_bytes.seek(0)
    saved_state = torch.load(state_bytes, weights_only=True)
    restored = quantize_model(_small_model(), bits=bits)
    restored.load_state_dict(saved_state)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens)["logits"], quantized_logits)

    dequantize_model(model)
    assert not hasattr(model, "_quant_bits")
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.layers[0].attention.q_proj, nn.Linear)


def test_quantize_invalid():
    with pytest.raises(ValueError, match="unsupported bits"):
        quantize_model(_small_model(), bits=7)


def test_int4_packs_odd_widths_and_supports_root_layers():
    model = nn.Linear(5, 3)
    tokens = torch.randn(2, 5)
    with torch.no_grad():
        baseline = model(tokens)

    quantized = quantize_model(model, bits=4)
    assert quantized.qweight.shape == (3, 3)
    with torch.no_grad():
        torch.testing.assert_close(quantized(tokens), baseline, rtol=0.2, atol=0.2)

    quantized = quantize_model(quantized, bits=8)
    assert quantized._quant_bits == 8
    restored = dequantize_model(quantized)
    assert isinstance(restored, nn.Linear)

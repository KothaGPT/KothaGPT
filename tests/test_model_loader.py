import pytest

from ml.inference.engine import KothaGPTEngine
from ml.inference.loader import load_model


def test_loader():
    model, tok, rec = load_model("kothagpt", device="cpu", quant="none")
    assert model is not None
    assert tok is not None
    assert rec.id == "kothagpt"
    assert rec.context_window == 8192


def test_loader_quant():
    model, _, _ = load_model("kothagpt-small", quant="int8")
    assert hasattr(model, "_quant_bits")


def test_loader_supports_bf16_mode():
    model, _, _ = load_model("kothagpt", quant="bf16")
    assert model is not None
    assert str(next(model.parameters()).dtype).startswith("torch.bfloat16")


def test_engine_uses_registry_loader_when_no_checkpoint_is_given():
    engine = KothaGPTEngine(model_id="kothagpt", device="cpu", quant="none")
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.context_window >= 4096
    generated = list(engine.generate("hello", max_new_tokens=2))
    assert generated


def test_loader_rejects_unsupported_quant_mode():
    with pytest.raises(ValueError, match="unsupported quant mode"):
        load_model("kothagpt", quant="fp16")


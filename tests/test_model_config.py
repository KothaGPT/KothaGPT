"""Tests for the base-model configuration system (WS-3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.models import (
    BaseModelConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    config_digest,
    load_config,
)


def test_defaults_validate(tmp_path: Path) -> None:
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"type": "bpe", "vocab": {f"t{i}": i for i in range(128)}}))
    cfg = BaseModelConfig(data=DataConfig(tokenizer_path=str(tok)))
    cfg.model.validate()
    cfg.training.validate()
    cfg.data.validate()


def test_intermediate_size_derived() -> None:
    cfg = ModelConfig(hidden_size=768)
    assert cfg.effective_intermediate_size() == int(8 / 3 * 768)


def test_intermediate_size_explicit_wins() -> None:
    cfg = ModelConfig(hidden_size=768, intermediate_size=2048)
    assert cfg.effective_intermediate_size() == 2048


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"architecture": "encoder_decoder"}, "architecture"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"num_layers": -1}, "num_layers"),
        ({"hidden_size": 768, "num_heads": 5}, "divisible"),
        ({"norm_type": "bogus"}, "norm_type"),
    ],
)
def test_model_config_validation_errors(kwargs, error: str) -> None:
    cfg = ModelConfig(**kwargs)
    with pytest.raises(ValueError, match=error):
        cfg.validate()


@pytest.mark.parametrize(
    "value, error",
    [
        ("fp32", "mixed_precision"),
        (0, "batch_size"),
    ],
)
def test_training_config_validation_errors(value, error: str) -> None:
    cfg = (
        TrainingConfig(mixed_precision=value)
        if error == "mixed_precision"
        else TrainingConfig(batch_size=value)
    )
    with pytest.raises(ValueError, match=error):
        cfg.validate()


def test_load_config_derives_vocab_size(tmp_path: Path) -> None:
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"type": "bpe", "vocab": {f"t{i}": i for i in range(777)}}))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"model:\n  hidden_size: 64\n  num_layers: 2\n  num_heads: 4\n"
        f"training:\n  max_steps: 5\n  warmup_steps: 1\n"
        f"data:\n  tokenizer_path: {tok}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.model.vocab_size == 777
    assert cfg.model.hidden_size == 64
    assert cfg.training.max_steps == 5


def test_load_config_explicit_vocab_size_wins(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "model:\n  vocab_size: 50\n  hidden_size: 64\n  num_layers: 2\n  num_heads: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.model.vocab_size == 50


def test_load_config_missing_tokenizer_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "model:\n  hidden_size: 64\n  num_layers: 2\n  num_heads: 4\n"
        "data:\n  tokenizer_path: /does/not/exist.json\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        load_config(cfg_path)


def test_load_small_yaml(tmp_path: Path) -> None:
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"type": "bpe", "vocab": {f"t{i}": i for i in range(50_000)}}))
    cfg = load_config("ml/configs/small.yaml")
    assert cfg.model.architecture == "decoder_transformer"
    assert cfg.model.vocab_size == 50_000
    assert cfg.training.mixed_precision == "bf16"
    assert cfg.data.train == "data/processed/train"
    assert cfg.training.gradient_accumulation_steps == 16


def test_config_digest_stable() -> None:
    a = BaseModelConfig(model=ModelConfig(hidden_size=64, num_layers=2, num_heads=4))
    b = BaseModelConfig(model=ModelConfig(hidden_size=64, num_layers=2, num_heads=4))
    assert config_digest(a) == config_digest(b)
    c = BaseModelConfig(model=ModelConfig(hidden_size=128, num_layers=2, num_heads=4))
    assert config_digest(a) != config_digest(c)

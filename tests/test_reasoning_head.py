from __future__ import annotations

import pytest
import torch

from ml.models.config import ModelConfig, config_digest
from ml.models.reasoning_head import ReasoningHead, ReasoningHeadLoss


def _config(**overrides) -> ModelConfig:
    options = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
    }
    options.update(overrides)
    return ModelConfig(**options)


def test_reasoning_head_output_shape():
    head = ReasoningHead(hidden_size=16, vocab_size=32, num_layers=2)
    hidden = torch.randn(2, 5, 16)
    logits = head(hidden)
    assert logits.shape == (2, 5, 32)


def test_reasoning_head_requires_3d_hidden_states():
    head = ReasoningHead(hidden_size=16, vocab_size=32)
    with pytest.raises(ValueError, match="3D"):
        head(torch.randn(2, 16))


def test_reasoning_head_rejects_invalid_dims():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        ReasoningHead(hidden_size=0, vocab_size=32)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ReasoningHead(hidden_size=16, vocab_size=0)
    with pytest.raises(ValueError, match="num_layers must be at least 1"):
        ReasoningHead(hidden_size=16, vocab_size=32, num_layers=0)
    with pytest.raises(ValueError, match="dropout must be in"):
        ReasoningHead(hidden_size=16, vocab_size=32, dropout=1.5)


def test_reasoning_head_loss_is_non_negative_and_backwardable():
    head = ReasoningHead(hidden_size=16, vocab_size=32, num_layers=2)
    hidden = torch.randn(2, 5, 16)
    logits = head(hidden)
    targets = torch.randint(0, 32, (2, 5))
    loss_fn = ReasoningHeadLoss()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    loss.backward()
    assert head.net[-1].weight.grad is not None


def test_reasoning_head_loss_respects_ignore_index():
    head = ReasoningHead(hidden_size=16, vocab_size=32)
    hidden = torch.randn(1, 3, 16)
    logits = head(hidden)
    targets = torch.tensor([[-100, 5, 7]])
    loss_fn = ReasoningHeadLoss()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0


def test_reasoning_head_loss_rejects_label_smoothing_out_of_range():
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        ReasoningHeadLoss(label_smoothing=1.5)


def test_reasoning_head_config_flag_changes_digest():
    base = _config()
    with_head = _config(reasoning_head=True)
    assert config_digest(base) != config_digest(with_head)


def test_reasoning_head_is_off_by_default():
    assert _config().reasoning_head is False
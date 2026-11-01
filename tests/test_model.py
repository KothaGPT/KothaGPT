"""Tests for the assembled KothaGPT causal-LM model (WS-5)."""

from __future__ import annotations

import pytest
import torch

from ml.models import KothaGPT
from ml.models.config import ModelConfig


def _cfg(**overrides) -> ModelConfig:
    params = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
    }
    params.update(overrides)
    return ModelConfig(**params)


def test_forward_shapes() -> None:
    model = KothaGPT(_cfg())
    input_ids = torch.randint(0, 64, (2, 8))
    out = model(input_ids=input_ids)
    assert out["logits"].shape == (2, 8, 64)


def test_forward_loss() -> None:
    model = KothaGPT(_cfg())
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    out = model(input_ids=input_ids, labels=labels)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_loss_ignores_masked_labels() -> None:
    model = KothaGPT(_cfg())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    labels[:, 4:] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        logits = model(input_ids=input_ids)["logits"]
    expected = torch.nn.functional.cross_entropy(
        logits[:, :4].reshape(-1, 64), labels[:, :4].reshape(-1)
    )
    assert torch.allclose(out["loss"], expected, atol=1e-5)


def test_learning_decreases_loss() -> None:
    torch.manual_seed(0)
    cfg = _cfg(vocab_size=4, hidden_size=16, num_layers=1, num_heads=2)
    model = KothaGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    seq = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        out = model(input_ids=seq[:, :-1], labels=seq[:, 1:])
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0]


def test_weight_tie() -> None:
    model = KothaGPT(_cfg(tie_word_embeddings=True))
    assert model.lm_head.weight is model.embed_tokens.weight


def test_no_weight_tie() -> None:
    model = KothaGPT(_cfg(tie_word_embeddings=False))
    assert model.lm_head.weight is not model.embed_tokens.weight


def test_num_parameters() -> None:
    cfg = _cfg()
    model = KothaGPT(cfg)
    assert model.num_parameters() > 0
    assert model.num_parameters(non_embedding=True) < model.num_parameters()


def test_gradient_checkpointing_parity() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    a = KothaGPT(cfg, gradient_checkpointing=False)
    b = KothaGPT(cfg, gradient_checkpointing=True)
    b.load_state_dict(a.state_dict())

    def step(model) -> float:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        optimizer.zero_grad()
        loss = model(input_ids=x, labels=y)["loss"]
        loss.backward()
        optimizer.step()
        return loss.item()

    torch.manual_seed(0)
    la = step(a)
    torch.manual_seed(0)
    lb = step(b)
    assert la == lb


def test_generate_length() -> None:
    model = KothaGPT(_cfg())
    model.eval()
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 8)


def test_generate_extends_past_max_context() -> None:
    cfg = _cfg(max_position_embeddings=6)
    model = KothaGPT(cfg)
    model.eval()
    long = torch.randint(0, 64, (1, 12))
    out = model.generate(long, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 15)
    assert torch.equal(out[:, : long.shape[1]], long)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"hidden_size": 32, "num_heads": 3}, "divisible"),
    ],
)
def test_invalid_config_raises(kwargs, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        KothaGPT(_cfg(**kwargs))

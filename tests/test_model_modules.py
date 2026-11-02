"""Tests for the base-model building blocks (WS-4)."""

from __future__ import annotations

import torch

from ml.models import (
    Attention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    apply_rotary_pos_emb,
    rotate_half,
)
from ml.models.config import ModelConfig


def _cfg(**overrides) -> ModelConfig:
    params = {
        "vocab_size": 32,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "max_position_embeddings": 16,
    }
    params.update(overrides)
    return ModelConfig(**params)


def test_rmsnorm_shape_and_scale_invariance() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16)
    y = norm(x)
    assert y.shape == x.shape
    scaled = norm(x * 3.0)
    assert torch.allclose(y, scaled, atol=1e-5)


def test_rmsnorm_gradient_flows() -> None:
    norm = RMSNorm(16)
    x = torch.randn(4, 16, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None
    assert norm.weight.grad is not None


def test_rotary_embedding_caches() -> None:
    emb = RotaryEmbedding(8, max_position_embeddings=16)
    cos, sin = emb(torch.empty(1, 1, 8), seq_len=16, device=torch.device("cpu"))
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_rotary_embedding_rotation_norm_preserving() -> None:
    emb = RotaryEmbedding(8, max_position_embeddings=32)
    q = torch.randn(1, 4, 32, 8)
    cos, sin = emb(q, seq_len=32, device=q.device)
    q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rotate_half() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    assert torch.allclose(rotated, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_swiglu_shape_and_grad() -> None:
    cfg = _cfg()
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
    y = mlp(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert mlp.down_proj.weight.grad is not None


def test_attention_causal_shape() -> None:
    cfg = _cfg()
    attn = Attention(cfg)
    x = torch.randn(2, 8, cfg.hidden_size)
    cos, sin = attn.rotary_emb(x, seq_len=8, device=x.device)
    y = attn(x, cos, sin)
    assert y.shape == (2, 8, cfg.hidden_size)


def test_attention_is_causal_position_0_sees_only_itself() -> None:
    cfg = _cfg()
    attn = Attention(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    cos, sin = attn.rotary_emb(x, seq_len=4, device=x.device)
    with torch.no_grad():
        y = attn(x, cos, sin)
        y0 = attn(x[:, :1], cos[:1], sin[:1])
    assert torch.isfinite(y).all()
    assert torch.allclose(y[:, 0, :], y0[:, 0, :], atol=1e-5)


def test_transformer_block_shape_and_residual() -> None:
    cfg = _cfg()
    block = TransformerBlock(cfg)
    x = torch.randn(2, 6, cfg.hidden_size)
    cos, sin = block.self_attn.rotary_emb(x, seq_len=6, device=x.device)
    y = block(x, cos, sin)
    assert y.shape == x.shape
    y.sum().backward()


def test_attention_gradient_flows() -> None:
    cfg = _cfg()
    attn = Attention(cfg)
    x = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    cos, sin = attn.rotary_emb(x, seq_len=4, device=x.device)
    attn(x, cos, sin).sum().backward()
    assert x.grad is not None
    for p in attn.parameters():
        assert p.grad is not None

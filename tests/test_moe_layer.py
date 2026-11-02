from __future__ import annotations

import pytest
import torch

from ml.models.moe.block import MoEBlock
from ml.models.moe.layer import ExpertFFN, SparseMoELayer


def test_expert_ffn_forward_and_backward():
    expert = ExpertFFN(hidden_size=8, intermediate_size=12)
    hidden = torch.randn(2, 3, 8, requires_grad=True)

    output = expert(hidden)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert expert.gate_proj.weight.grad is not None


def test_sparse_moe_forward_routes_experts_and_backpropagates_auxiliary_loss():
    torch.manual_seed(7)
    layer = SparseMoELayer(
        hidden_size=8,
        num_experts=4,
        active_experts=2,
        intermediate_size=12,
    )
    hidden = torch.randn(2, 3, 8, requires_grad=True)

    output = layer(hidden)
    auxiliary_loss = layer.auxiliary_loss()
    (output.square().mean() + 0.1 * auxiliary_loss).backward()

    assert output.shape == hidden.shape
    assert auxiliary_loss.ndim == 0
    assert auxiliary_loss.requires_grad
    assert torch.isfinite(auxiliary_loss)
    assert hidden.grad is not None
    assert layer.router.gate.weight.grad is not None
    assert layer.router.gate.weight.grad.abs().sum() > 0
    expert_gradients = [expert.gate_proj.weight.grad for expert in layer.experts]
    assert any(gradient is not None for gradient in expert_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in expert_gradients
        if gradient is not None
    )


def test_sparse_moe_preserves_arbitrary_leading_dimensions():
    layer = SparseMoELayer(8, num_experts=3, active_experts=1, intermediate_size=10)

    assert layer(torch.randn(2, 4, 5, 8)).shape == (2, 4, 5, 8)


def test_auxiliary_loss_requires_forward():
    layer = SparseMoELayer(8, num_experts=3, active_experts=2, intermediate_size=10)

    with pytest.raises(RuntimeError, match="forward must be called"):
        layer.auxiliary_loss()


def test_moe_block_integrates_attention_experts_and_auxiliary_loss():
    block = MoEBlock(
        hidden_size=16,
        num_experts=3,
        active_experts=2,
        intermediate_size=24,
        num_heads=4,
    )
    hidden = torch.randn(2, 5, 16, requires_grad=True)

    output = block(hidden)
    loss = output.square().mean() + 0.01 * block.auxiliary_loss()
    loss.backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert block.feed_forward.router.gate.weight.grad is not None
    assert block.attention.q_proj.weight.grad is not None

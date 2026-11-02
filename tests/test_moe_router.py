from __future__ import annotations

import pytest
import torch

from ml.models.moe.router import Router


def test_router_selects_top_k_experts_and_normalizes_weights():
    router = Router(hidden_size=3, num_experts=4, active_experts=2)
    with torch.no_grad():
        router.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )
    hidden = torch.tensor([[[4.0, 2.0, 1.0], [1.0, 3.0, 2.0]]])

    indices, weights = router.route(hidden)

    assert indices.shape == (2, 2)
    assert indices.tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert torch.all(weights[:, 0] >= weights[:, 1])


def test_router_top_k_weights_and_auxiliary_loss_backpropagate_to_gate():
    torch.manual_seed(3)
    router = Router(hidden_size=4, num_experts=3, active_experts=2)
    hidden = torch.randn(2, 5, 4)

    indices, weights = router(hidden)
    loss = weights.square().sum() + router.auxiliary_loss()
    loss.backward()

    assert indices.shape == (10, 2)
    assert weights.requires_grad
    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()
    assert router.gate.weight.grad.abs().sum() > 0


def test_router_auxiliary_loss_requires_routing_first():
    router = Router(hidden_size=4, num_experts=3)

    with pytest.raises(RuntimeError, match="route must be called"):
        router.auxiliary_loss()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0, "num_experts": 2}, "hidden_size"),
        ({"hidden_size": 4, "num_experts": 0}, "num_experts"),
        ({"hidden_size": 4, "num_experts": 2, "active_experts": 3}, "active_experts"),
        ({"hidden_size": 4, "num_experts": 2, "capacity_factor": 0}, "capacity_factor"),
    ],
)
def test_router_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Router(**kwargs)

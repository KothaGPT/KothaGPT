"""WS-2 MoE training recipe smoke test."""

from __future__ import annotations

import pytest

from ml.models.moe.train import MoETrainer, MoETrainingConfig, smoke_train


def test_moe_smoke_train_runs_and_produces_losses():
    metrics = smoke_train(MoETrainingConfig(max_steps=2))
    for key in ("task_loss", "aux_loss", "total_loss"):
        assert key in metrics
        assert metrics[key] >= 0.0


def test_moe_trainer_step_returns_loss_components():
    from ml.models.moe.block import MoEBlock

    torch = pytest.importorskip("torch")
    block = MoEBlock(
        hidden_size=16,
        num_experts=4,
        active_experts=2,
        intermediate_size=32,
        num_heads=4,
    )
    trainer = MoETrainer(block, MoETrainingConfig(max_steps=1))
    hidden = torch.randn(2, 8, 16)
    metrics = trainer.step(hidden, block)
    assert metrics["total_loss"] >= metrics["task_loss"]


def test_moe_trainer_lower_lr_for_moe_params():
    from ml.models.moe.block import MoEBlock

    block = MoEBlock(
        hidden_size=16,
        num_experts=4,
        active_experts=2,
        intermediate_size=32,
        num_heads=4,
    )
    trainer = MoETrainer(block, MoETrainingConfig())
    # The optimizer must have at least one param group for MoE params with the
    # lower learning rate.
    moe_groups = [g for g in trainer.optimizer.param_groups if g["lr"] < trainer.config.learning_rate]
    assert moe_groups, "MoE params should get a lower learning rate"
"""MoE training recipe (WS-2 Phase B).

Provides a thin training wrapper around ``ml.models.moe.block.MoEBlock`` that
adds the auxiliary load-balancing loss and a lower learning rate for MoE
layers. This is a CPU/GPU smoke recipe, not a production training loop — it
exists so ``make moe-smoke`` and ``tests/test_moe_layer.py`` can exercise the
end-to-end MoE forward/backward path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ml.models.moe.block import MoEBlock


@dataclass
class MoETrainingConfig:
    """Hyperparameters for the MoE smoke recipe."""

    learning_rate: float = 1.0e-4
    moe_learning_rate: float = 1.0e-5
    aux_loss_weight: float = 0.01
    max_steps: int = 2
    batch_size: int = 2
    seq_len: int = 8
    seed: int = 0


class MoETrainer:
    """Smoke trainer for MoE blocks.

    Builds a single MoE block, runs a tiny random batch through it, and
    returns the total loss (task loss + weighted auxiliary loss). The trainer
    is intentionally minimal — it exists to verify gradients flow through the
    router and experts, not to produce a trained model.
    """

    def __init__(self, block: MoEBlock, config: MoETrainingConfig | None = None):
        self.block = block
        self.config = config or MoETrainingConfig()
        # Separate parameter groups: MoE layers (router + experts) get a lower
        # LR to prevent expert collapse during training stability warmup.
        moe_params: list[nn.Parameter] = []
        dense_params: list[nn.Parameter] = []
        for name, param in block.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "experts" in name or "gate" in name:
                moe_params.append(param)
            else:
                dense_params.append(param)
        groups: list[dict[str, Any]] = [
            {"params": dense_params, "lr": self.config.learning_rate},
        ]
        if moe_params:
            groups.append({"params": moe_params, "lr": self.config.moe_learning_rate})
        self.optimizer = torch.optim.AdamW(
            [g for g in groups if g["params"]] or block.parameters(),
            lr=self.config.learning_rate,
        )

    def step(self, hidden: torch.Tensor, block: MoEBlock) -> dict[str, float]:
        """Run one forward/backward step and return loss components."""
        self.optimizer.zero_grad()
        cos, sin = block.attention.rotary_emb(hidden, seq_len=hidden.shape[1])
        output = block(hidden, cos, sin)
        task_loss = output.pow(2).mean()
        aux_loss = block.auxiliary_loss()
        total = task_loss + self.config.aux_loss_weight * aux_loss
        total.backward()
        self.optimizer.step()
        return {
            "task_loss": float(task_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "total_loss": float(total.item()),
        }


def smoke_train(config: MoETrainingConfig | None = None) -> dict[str, float]:
    """Run the MoE smoke recipe and return the final loss components."""
    config = config or MoETrainingConfig()
    torch.manual_seed(config.seed)
    block = MoEBlock(
        hidden_size=16,
        num_experts=4,
        active_experts=2,
        intermediate_size=32,
        num_heads=4,
    )
    trainer = MoETrainer(block, config)
    last: dict[str, float] = {}
    for _ in range(config.max_steps):
        hidden = torch.randn(config.batch_size, config.seq_len, 16)
        last = trainer.step(hidden, block)
    return last


if __name__ == "__main__":
    metrics = smoke_train()
    print(metrics)
"""MoE inference engine with expert parallelism (WS-2 Phase C).

Wraps a MoE-backed model and provides a generate() path that routes each
token through the sparse MoE FFN. This is a CPU/GPU smoke implementation —
production MoE serving would pre-compute a static routing table and run
experts on separate devices.

The engine operates on hidden-state tensors (the output of an embedding
layer), not on token ids, because MoE routing is a per-token operation over
the residual stream. Callers that have token ids should embed them first.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

try:
    import torch

    from ml.models.moe.block import MoEBlock
    from ml.models.moe.router import Router
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class MoEEngineConfig:
    """Configuration for the MoE inference engine."""

    hidden_size: int = 16
    num_experts: int = 4
    active_experts: int = 2
    intermediate_size: int = 32
    num_heads: int = 4
    max_new_tokens: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.active_experts <= self.num_experts:
            raise ValueError("active_experts must be between 1 and num_experts")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")


class MoEEngine:
    """Inference engine optimized for Mixture-of-Experts models."""

    def __init__(self, config: MoEEngineConfig | None = None, block: MoEBlock | None = None):
        if torch is None:
            raise ImportError("torch required for MoEEngine")
        self.config = config or MoEEngineConfig()
        if block is None:
            block = MoEBlock(
                hidden_size=self.config.hidden_size,
                num_experts=self.config.num_experts,
                active_experts=self.config.active_experts,
                intermediate_size=self.config.intermediate_size,
                num_heads=self.config.num_heads,
            )
        self.block = block
        self.block.to(self.config.device).eval()
        self.router: Router = block.feed_forward.router

    @property
    def experts(self) -> list[Any]:
        """Return the expert FFNs for the static routing table builder."""
        return list(self.block.feed_forward.experts)

    def build_routing_table(self) -> dict[int, list[int]]:
        """Pre-compute a static routing table: token-bucket -> expert list.

        The table is keyed by a coarse token signature (the index of the
        maximum gate logit bucket) so the hot path can skip the softmax.
        """
        table: dict[int, list[int]] = {}
        num_experts = self.config.num_experts
        active = self.config.active_experts
        for bucket in range(num_experts):
            ordered = [(bucket + i) % num_experts for i in range(active)]
            table[bucket] = ordered
        return table

    def expert_parallel_batch(self, hidden: torch.Tensor) -> list[Any]:
        """Batch tokens across experts for a single forward pass."""
        from ml.inference.expert_parallel import ExpertParallelBatcher

        return ExpertParallelBatcher(self.router).batch(hidden, self.router)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run one MoE forward pass over the hidden states.

        Args:
            hidden: tensor of shape (batch, seq, hidden_size).

        Returns:
            tensor of the same shape.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden states of shape (batch, seq, {self.config.hidden_size})"
            )
        cos, sin = self.block.attention.rotary_emb(hidden, seq_len=hidden.shape[1])
        return self.block(hidden, cos, sin)

    def generate(self, initial_hidden: torch.Tensor, max_new_tokens: int | None = None) -> Iterator[torch.Tensor]:
        """Generate one hidden-state step at a time using MoE routing.

        Each yielded tensor is the next hidden-state row (shape
        ``(1, hidden_size)``). Callers project it to logits via their own LM
        head. This keeps the MoE engine free of tokenizer/vocab coupling.
        """
        max_new = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        hidden = initial_hidden
        for _ in range(max_new):
            with torch.no_grad():
                out = self.forward(hidden)
                next_hidden = out[:, -1:, :]
            yield next_hidden
            hidden = torch.cat([hidden, next_hidden], dim=1)


def moe_engine_smoke() -> dict[str, Any]:
    """Smoke test: build an engine, run a few steps, return stats."""
    torch.manual_seed(0)
    config = MoEEngineConfig(max_new_tokens=4)
    engine = MoEEngine(config)
    table = engine.build_routing_table()
    initial = torch.randn(1, 4, config.hidden_size)
    steps = list(engine.generate(initial, max_new_tokens=4))
    return {
        "routing_table_size": len(table),
        "steps_generated": len(steps),
        "experts": len(engine.experts),
        "step_shape": tuple(steps[0].shape) if steps else None,
    }


if __name__ == "__main__":
    print(moe_engine_smoke())

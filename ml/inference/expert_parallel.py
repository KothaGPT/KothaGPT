"""Expert-parallel batching for MoE inference (WS-2 Phase C).

Provides a lightweight batching helper that groups tokens by their assigned
expert so a single forward pass can dispatch work to the right experts. This
is a CPU/GPU smoke implementation — production expert parallelism would run
experts on separate devices and stream tokens across them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from ml.models.moe.router import Router


@dataclass
class ExpertBatch:
    """Tokens assigned to one expert, with their routing weights."""

    expert_index: int
    token_indices: list[int]
    weights: list[float]
    hidden: torch.Tensor


class ExpertParallelBatcher:
    """Batch tokens across experts for a single MoE forward pass.

    Given a batch of hidden states and a router, produces one ``ExpertBatch``
    per non-empty expert so the caller can run each expert's FFN in isolation
    and scatter the weighted outputs back into the output tensor.
    """

    def __init__(self, router: Router | None = None):
        self.router = router

    def batch(self, hidden: torch.Tensor, router: Router | None = None) -> list[ExpertBatch]:
        """Return per-expert batches for the given hidden states."""
        active = router or self.router
        if active is None:
            raise ValueError("a router is required to batch by expert")
        if hidden.ndim < 2 or hidden.shape[-1] != active.hidden_size:
            raise ValueError(
                f"expected hidden states with final dimension {active.hidden_size}"
            )
        tokens = hidden.reshape(-1, active.hidden_size)
        if tokens.shape[0] == 0:
            raise ValueError("cannot batch an empty token batch")
        indices, weights = active.route(tokens)
        groups: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"token_indices": [], "weights": [], "rows": []}
        )
        for token_row, (expert_row, weight_row) in enumerate(
            zip(indices.tolist(), weights.tolist())
        ):
            for expert_index, weight in zip(expert_row, weight_row):
                groups[expert_index]["token_indices"].append(token_row)
                groups[expert_index]["weights"].append(float(weight))
                groups[expert_index]["rows"].append(token_row)
        batches: list[ExpertBatch] = []
        for expert_index in sorted(groups):
            g = groups[expert_index]
            rows = torch.tensor(g["rows"], dtype=torch.long, device=tokens.device)
            batches.append(
                ExpertBatch(
                    expert_index=expert_index,
                    token_indices=g["token_indices"],
                    weights=g["weights"],
                    hidden=tokens.index_select(0, rows),
                )
            )
        return batches


def expert_parallel_smoke() -> dict[str, int]:
    """Smoke test: batch random tokens across 4 experts and count assignments."""
    torch.manual_seed(0)
    router = Router(hidden_size=16, num_experts=4, active_experts=2)
    batcher = ExpertParallelBatcher(router)
    hidden = torch.randn(8, 16)
    batches = batcher.batch(hidden, router)
    total_tokens = sum(len(b.token_indices) for b in batches)
    return {"experts_used": len(batches), "tokens_routed": total_tokens}


if __name__ == "__main__":
    print(expert_parallel_smoke())

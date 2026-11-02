"""Sparse mixture-of-experts feed-forward layer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ml.models.moe.router import Router


class ExpertFFN(nn.Module):
    """SwiGLU feed-forward network used by one expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SparseMoELayer(nn.Module):
    """Top-k token routing over SwiGLU experts with a load-balancing loss."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        active_experts: int,
        intermediate_size: int,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        if not math.isfinite(capacity_factor) or capacity_factor <= 0:
            raise ValueError("capacity_factor must be a finite positive number")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.intermediate_size = intermediate_size
        self.capacity_factor = capacity_factor
        self.router = Router(
            hidden_size,
            num_experts,
            active_experts,
            capacity_factor,
        )
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size) for _ in range(num_experts)
        )
        self._auxiliary_loss: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden states with final dimension {self.hidden_size}"
            )
        original_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.hidden_size)
        expert_indices, routing_weights = self.router.route(hidden_states)
        output = torch.zeros_like(tokens)

        for expert_index, expert in enumerate(self.experts):
            token_indices, top_k_slots = torch.where(expert_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(tokens.index_select(0, token_indices))
            weights = routing_weights[token_indices, top_k_slots].unsqueeze(-1)
            output = output.index_add(
                0,
                token_indices,
                expert_output * weights.to(dtype=expert_output.dtype),
            )

        self._auxiliary_loss = self.router.auxiliary_loss()
        return output.reshape(original_shape)

    def auxiliary_loss(self) -> torch.Tensor:
        """Return the load-balancing loss from the most recent forward pass."""
        if self._auxiliary_loss is None:
            raise RuntimeError("forward must be called before auxiliary_loss")
        return self._auxiliary_loss

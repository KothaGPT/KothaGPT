"""Differentiable top-k routing and load-balancing loss for MoE layers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Router(nn.Module):
    """Learned softmax gate with sparse top-k token-to-expert assignments."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        active_experts: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= active_experts <= num_experts:
            raise ValueError("active_experts must be between 1 and num_experts")
        if not math.isfinite(capacity_factor) or capacity_factor <= 0:
            raise ValueError("capacity_factor must be a finite positive number")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self._last_probabilities: torch.Tensor | None = None
        self._last_indices: torch.Tensor | None = None

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened-token expert indices and differentiable top-k weights."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden states with final dimension {self.hidden_size}"
            )
        tokens = hidden_states.reshape(-1, self.hidden_size)
        if tokens.shape[0] == 0:
            raise ValueError("cannot route an empty token batch")

        logits = self.gate(tokens)
        probabilities = F.softmax(logits.float(), dim=-1)
        top_probabilities, indices = probabilities.topk(self.active_experts, dim=-1)
        weights = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
        self._last_probabilities = probabilities
        self._last_indices = indices
        return indices, weights.to(dtype=logits.dtype)

    def auxiliary_loss(self) -> torch.Tensor:
        """Switch-style load balancing loss with gradients through gate probabilities."""
        if self._last_probabilities is None or self._last_indices is None:
            raise RuntimeError("route must be called before auxiliary_loss")
        probabilities = self._last_probabilities
        assignments = F.one_hot(self._last_indices, self.num_experts).float().sum(dim=1)
        tokens = assignments.shape[0]
        importance = probabilities.mean(dim=0)
        load = assignments.sum(dim=0) / (tokens * self.active_experts)
        return self.num_experts * torch.sum(importance * load)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.route(hidden_states)

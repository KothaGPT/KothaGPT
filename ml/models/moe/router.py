"""Router implementation for Mixture-of-Experts."""

from __future__ import annotations


class Router:
    """Softmax gating router with top-k expert selection."""

    def __init__(self, hidden_size: int, num_experts: int,
                 active_experts: int = 2, capacity_factor: float = 1.25):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.capacity_factor = capacity_factor

    def route(self, hidden_states):
        """Route tokens to top-k experts. Returns (indices, weights)."""
        raise NotImplementedError

    def auxiliary_loss(self) -> float:
        """Load-balancing auxiliary loss to prevent expert collapse."""
        raise NotImplementedError

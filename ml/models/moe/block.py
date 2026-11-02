"""Transformer block with Mixture-of-Experts FFN."""

from __future__ import annotations


class MoEBlock:
    """Transformer block replacing dense FFN with sparse MoE."""

    def __init__(self, hidden_size: int, num_experts: int,
                 active_experts: int, intermediate_size: int,
                 num_heads: int, rope_theta: float = 500000.0):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.rope_theta = rope_theta

    def forward(self, hidden_states):
        """Forward pass through attention + MoE FFN."""
        raise NotImplementedError

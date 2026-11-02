"""Transformer block using sparse MoE feed-forward experts."""

from __future__ import annotations

import torch
from torch import nn

from ml.models.config import ModelConfig
from ml.models.layers import Attention, RMSNorm
from ml.models.moe.layer import SparseMoELayer


class MoEBlock(nn.Module):
    """Pre-norm transformer block with attention and a sparse MoE FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        active_experts: int,
        intermediate_size: int,
        num_heads: int,
        rope_theta: float = 500000.0,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        config = ModelConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=1,
            max_position_embeddings=4096,
            rope_theta=rope_theta,
        )
        config.validate()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.feed_forward = SparseMoELayer(
            hidden_size,
            num_experts,
            active_experts,
            intermediate_size,
            capacity_factor,
        )

    @property
    def self_attn(self) -> Attention:
        return self.attention

    @property
    def moe(self) -> SparseMoELayer:
        return self.feed_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states), cos, sin
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

    def auxiliary_loss(self) -> torch.Tensor:
        return self.feed_forward.auxiliary_loss()

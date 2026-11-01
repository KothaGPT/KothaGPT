from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        theta: float | None = None,
        scaling: str | None = None,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base if theta is None else theta
        self.scaling = scaling
        self.scaling_factor = scaling_factor
        self.inv_freq = self._compute_inv_freq()
        self._build_cache(max_position_embeddings)

    def _effective_base(self) -> float:
        base = self.base
        if self.scaling == "ntk":
            # NTK-aware scaling: widen frequencies by scaling_factor^(dim/(dim-2)).
            base = base * self.scaling_factor ** (self.dim / (self.dim - 2))
        return base

    def _compute_inv_freq(self) -> torch.Tensor:
        base = self._effective_base()
        return 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def _scaled_positions(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        if self.scaling == "linear":
            positions = positions / self.scaling_factor
        return positions

    def _build_cache(self, seq_len: int) -> None:
        positions = self._scaled_positions(seq_len)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None, device: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1] if x.dim() >= 2 else x.shape[0]
        if seq_len <= self.max_position_embeddings:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        positions = self._scaled_positions(seq_len)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
            scaling_factor=config.rope_scaling_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if cos is None or sin is None:
            cos, sin = self.rotary_emb(x, seq_len=T)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate = config.effective_intermediate_size()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def self_attn(self) -> Attention:
        return self.attention

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin)
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


class KothaGPT(nn.Module):
    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        config.validate()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def rotary_emb(self) -> RotaryEmbedding:
        return self.layers[0].attention.rotary_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        hidden = self.embed_tokens(input_ids)

        cos, sin = self.layers[0].attention.rotary_emb(hidden, seq_len=T)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(layer, hidden, cos, sin)
            else:
                hidden = layer(hidden, cos, sin)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                generated = input_ids
                max_positions = self.config.max_position_embeddings
                for _ in range(max_new_tokens):
                    context = generated[:, -max_positions:]
                    logits = self(context)["logits"][:, -1, :]
                    if temperature > 0:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)
                return generated
        finally:
            self.train(was_training)

    def num_parameters(self, non_embedding: bool = False) -> int:
        if non_embedding:
            return sum(
                p.numel()
                for name, p in self.named_parameters()
                if "embed_tokens" not in name
            )
        return sum(p.numel() for p in self.parameters())

    def extend_context_length(
        self,
        new_length: int,
        *,
        rope_theta: float | None = None,
        rope_scaling: str | None = None,
        rope_scaling_factor: float = 1.0,
    ) -> None:
        """Extend the model's context window via RoPE scaling.

        Parameter weights are preserved exactly; only the rotary cache and
        ``config`` are updated. Shrinking or staying the same is rejected.
        """
        if new_length <= self.config.max_position_embeddings:
            raise ValueError(
                f"new_length {new_length} must exceed current "
                f"max_position_embeddings {self.config.max_position_embeddings}"
            )
        if rope_theta is not None:
            self.config.rope_theta = rope_theta
        if rope_scaling is not None:
            self.config.rope_scaling = rope_scaling
        if rope_scaling_factor is not None:
            self.config.rope_scaling_factor = rope_scaling_factor
        self.config.max_position_embeddings = new_length
        for layer in self.layers:
            emb = layer.attention.rotary_emb
            emb.max_position_embeddings = new_length
            emb.scaling = self.config.rope_scaling
            emb.scaling_factor = self.config.rope_scaling_factor
            emb.base = self.config.rope_theta
            emb.inv_freq = emb._compute_inv_freq()
            emb._build_cache(new_length)

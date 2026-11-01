"""WS-2 KV cache — per-layer key/value cache for autoregressive generation."""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, num_layers: int, max_batch: int = 1, max_seq: int = 4096, n_heads: int = 4, head_dim: int = 32):
        self.num_layers = num_layers
        self.cache: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None) for _ in range(num_layers)]
        self.max_seq = max_seq
        self.n_heads = n_heads
        self.head_dim = head_dim

    def get(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.cache[layer]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # k,v: [B, H, T, D]
        prev_k, prev_v = self.cache[layer]
        if prev_k is None:
            self.cache[layer] = (k, v)
        else:
            self.cache[layer] = (torch.cat([prev_k, k], dim=2), torch.cat([prev_v, v], dim=2))
        return self.cache[layer]

    def clear(self) -> None:
        self.cache = [(None, None) for _ in range(self.num_layers)]

    def seq_len(self, layer: int = 0) -> int:
        k, _ = self.cache[layer]
        return 0 if k is None else k.shape[2]

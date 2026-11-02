"""Reward model — scalar head on KothaGPT (WS-2)."""

from __future__ import annotations

import torch
from torch import nn

from ml.models import KothaGPT


class RewardModel(nn.Module):
    def __init__(self, base: KothaGPT):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Use base model's last hidden state (mean pool)
        out = self.base(input_ids, output_hidden_states=True)
        # base returns dict with hidden_states or logits; fallback to logits mean
        if isinstance(out, dict) and "hidden_states" in out:
            h = out["hidden_states"][-1]  # [B, T, H]
        else:
            # approximate via embeddings
            h = self.base.get_input_embeddings()(input_ids)
        # Mean pool over non-masked tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.head(h).squeeze(-1)  # [B]

    def score_pair(self, tokenizer, prompt: str, completion: str, device: str = "cpu") -> float:
        # Ensure tensor is on the same device as the reward model (covers self.base + self.head)
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = torch.device(device)
        else:
            # Move entire model if requested device differs
            if str(model_device) != device:
                self.to(device)
                model_device = torch.device(device)
        ids = tokenizer.encode(prompt + completion)[:512]
        inp = torch.tensor([ids], device=model_device)
        with torch.no_grad():
            return float(self.forward(inp).item())

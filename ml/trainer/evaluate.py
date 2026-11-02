"""Held-out evaluation: perplexity + greedy sample generation."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from ml.models import KothaGPT
from ml.tokenizer import BaseTokenizer

from .dataset import CausalLMDataset


@torch.no_grad()
def evaluate(
    model: KothaGPT,
    dataset: CausalLMDataset,
    *,
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "cpu",
) -> dict:
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0.0
    seen = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        total += model(input_ids=input_ids, labels=labels)["loss"].item()
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    loss = total / max(seen, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 20))}


@torch.no_grad()
def sample_text(
    model: KothaGPT,
    tokenizer: BaseTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    model.to(device)
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.numel() == 0:
        ids = torch.tensor([[tokenizer.unk_id]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())

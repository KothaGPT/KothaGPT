"""SFT trainer wrapping ml.trainer loop (WS-2)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ml.instruction.dataset import InstructionCollator
from ml.models import KothaGPT
from ml.sft.dataset import SFTDataset


def run_sft_trainer(
    model: KothaGPT,
    records,
    tokenizer,
    *,
    device: str = "cpu",
    max_steps: int = 20,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    template: str = "default",
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    dataset = SFTDataset(records, tokenizer, max_length, template=template)
    pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=InstructionCollator(pad_id, max_length),
    )

    # Filter dataset to ensure every retained record has completion tokens within max_length
    # (SFTDataset already raises for prompt-only, but collator could still produce all -100 if truncated)
    def _has_completion(batch) -> bool:
        labs = batch["labels"]
        return (labs != -100).any().item()

    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, total = 0, 0.0
    initial_loss = None
    last_loss: float | None = None
    while steps < max_steps:
        progressed = False
        for batch in loader:
            if not _has_completion(batch):
                continue
            progressed = True
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device)
            )
            loss = outputs["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"non-finite loss at step {steps}: {loss.item()}")
            if initial_loss is None:
                initial_loss = float(loss.detach())
            last_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            total += last_loss
            steps += 1
            if steps >= max_steps:
                break
        if not progressed:
            break
    avg_loss = total / max(steps, 1)
    if steps == 0:
        raise ValueError("SFT training produced no batches with supervised completion tokens")
    return {
        "steps": steps,
        "loss": avg_loss,
        "initial_loss": initial_loss,
        "final_loss": last_loss if last_loss is not None else avg_loss,
        "mean_loss": avg_loss,
    }

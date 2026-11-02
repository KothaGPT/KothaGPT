"""Completion-only supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.models import KothaGPT, load_config
from ml.tokenizer import load_tokenizer

from .dataset import InstructionCollator, InstructionDataset, load_jsonl


def run_sft(
    model: KothaGPT,
    records,
    tokenizer,
    *,
    device="cpu",
    max_steps=1,
    batch_size=1,
    learning_rate=3e-4,
    max_length=512,
):
    """Run a small, deterministic completion-only SFT loop and return metrics."""
    dataset = InstructionDataset(records, tokenizer, max_length)
    pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=InstructionCollator(pad_id, max_length),
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, total = 0, 0.0
    while steps < max_steps:
        progressed = False
        for batch in loader:
            progressed = True
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device)
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            if steps >= max_steps:
                break
        if not progressed:
            break
    return {"steps": steps, "loss": total / max(steps, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run completion-only SFT")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="ml/sft/artifacts/run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    records = load_jsonl(args.train)
    metrics = run_sft(
        KothaGPT(config.model),
        records,
        tokenizer,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=config.model.max_position_embeddings,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(
        __import__("json").dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics)


if __name__ == "__main__":
    main()

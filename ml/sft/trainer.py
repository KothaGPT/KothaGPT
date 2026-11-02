"""SFT trainer wrapping ml.trainer loop (WS-2)."""

from __future__ import annotations

import json
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ml.models import BaseModelConfig, DataConfig, KothaGPT, TrainingConfig
from ml.sft.dataset import SFTDataset
from ml.trainer.checkpoint import latest_checkpoint, step_of
from ml.trainer.loop import train


class _PaddedSFTDataset:
    def __init__(self, dataset: SFTDataset, max_length: int, pad_id: int):
        self.dataset = dataset
        self.max_length = max_length
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[index]
        input_ids = item["input_ids"][: self.max_length]
        labels = item["labels"][: self.max_length]
        padding = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_id] * padding
        labels = labels + [-100] * padding
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


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
    config: BaseModelConfig | None = None,
    out_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)
    dataset = _PaddedSFTDataset(
        SFTDataset(records, tokenizer, max_length, template=template), max_length, pad_id
    )

    if config is None:
        training = TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            warmup_steps=0,
            min_lr=learning_rate / 10,
            mixed_precision="none",
        )
        config = BaseModelConfig(
            model=model.config,
            training=training,
            data=DataConfig(train="sft-records", tokenizer_path=""),
        )
    config = replace(
        config,
        training=replace(config.training, batch_size=batch_size, learning_rate=learning_rate),
    )

    temporary_directory = tempfile.TemporaryDirectory() if out_dir is None else nullcontext(None)
    with temporary_directory as temporary_path:
        run_dir = Path(temporary_path) if temporary_path else Path(out_dir)
        if not config.data.tokenizer_path:
            tokenizer_path = tokenizer.save(run_dir / "tokenizer")
            config = replace(
                config,
                data=replace(config.data, tokenizer_path=str(tokenizer_path.resolve())),
            )

        start_step = 0
        if resume_from is not None:
            checkpoint = latest_checkpoint(resume_from)
            if checkpoint is not None:
                start_step = step_of(checkpoint)

        result = train(
            model,
            config,
            dataset,
            None,
            out_dir=run_dir,
            device=device,
            max_steps=max_steps,
            resume_from=resume_from,
        )

        history_path = run_dir / "history.jsonl"
        losses = [
            row["loss"]
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
            if row.get("step", 0) > start_step and "loss" in row
        ]
        if not losses:
            raise ValueError("SFT training produced no batches with supervised completion tokens")
        mean_loss = sum(losses) / len(losses)
        return {
            "steps": result["step"] - start_step,
            "step": result["step"],
            "loss": result["loss"],
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": mean_loss,
        }

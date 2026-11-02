"""SFT dataset mixing and assistant-masked batching (WS-2).

Wraps ml.instruction.dataset with mixing weights and chat-template awareness.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ml.instruction.dataset import (
    InstructionDataset,
    InstructionRecord,
    load_jsonl,
)
from ml.sft.templates import apply_chat_template

__all__ = ["SFTDataset", "SFTMix", "load_mix"]


class SFTMix:
    """Builds a weighted mixture across task types/sources with contamination check."""

    def __init__(
        self,
        records: list[InstructionRecord],
        weights: dict[str, float] | None = None,
        seed: int = 0,
    ):
        self.records = records
        self.weights = weights or {}
        self.seed = seed

    def build(self) -> list[InstructionRecord]:
        if not self.weights:
            return list(self.records)
        # Validate weights
        import math

        for cat, w in self.weights.items():
            if not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
                raise ValueError(f"weight for {cat!r} must be finite non-negative, got {w!r}")
        by_cat: dict[str, list[InstructionRecord]] = {}
        for r in self.records:
            by_cat.setdefault(r.category, []).append(r)
        total_weight = sum(self.weights.get(cat, 1.0) for cat in by_cat)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(f"total weight must be finite positive, got {total_weight}")
        rng = random.Random(self.seed)
        mixed: list[InstructionRecord] = []
        for cat, recs in by_cat.items():
            w = self.weights.get(cat, 1.0)
            # Normalized weight without max floor
            target = int(len(self.records) * w / total_weight)
            if target <= 0:
                continue
            if len(recs) >= target:
                mixed.extend(rng.sample(recs, target))
            else:
                for _ in range(target):
                    mixed.append(rng.choice(recs))
        rng.shuffle(mixed)
        return mixed

    def check_contamination(self, eval_records: list[InstructionRecord]) -> list[str]:
        """Return evaluation digests found in training (stable prompt fingerprint)."""
        import hashlib
        import json

        def _fingerprint(r: InstructionRecord) -> str:
            # Canonical prompt fields: messages + input
            payload = {
                "messages": [{"role": m["role"], "content": m["content"]} for m in r.messages]
                if r.messages
                else [],
                "input": r.input,
                "instruction": r.instruction,
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()

        train_fps = {_fingerprint(r) for r in self.records}
        return [_fingerprint(r) for r in eval_records if _fingerprint(r) in train_fps]


def load_mix(
    paths: list[str | Path], weights: dict[str, float] | None = None
) -> list[InstructionRecord]:
    records: list[InstructionRecord] = []
    for p in paths:
        records.extend(load_jsonl(p))
    return SFTMix(records, weights).build()


class SFTDataset(InstructionDataset):
    """InstructionDataset with chat-template and EOS-aware masking.

    Uses templates.apply_chat_template for prompt construction when messages present,
    otherwise falls back to InstructionRecord.prompt().
    """

    def __init__(
        self,
        records: Iterable[InstructionRecord],
        tokenizer,
        max_length: int,
        template: str = "default",
    ):
        super().__init__(records, tokenizer, max_length)
        self.template = template

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.messages:
            # Retain all messages except the final assistant that equals record.output
            msgs = list(record.messages)
            # Find last assistant that matches output and exclude only that one
            last_idx = None
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "assistant" and msgs[i]["content"] == record.output:
                    last_idx = i
                    break
            if last_idx is not None:
                prompt_msgs = msgs[:last_idx] + msgs[last_idx + 1 :]
                # Keep only non-assistant before last? Actually keep all before last including earlier assistants
                prompt_msgs = [m for j, m in enumerate(msgs) if j != last_idx]
            else:
                prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            # Remove trailing assistant if still present (should not)
            prompt_msgs = [
                m
                for m in prompt_msgs
                if not (m["role"] == "assistant" and m["content"] == record.output)
            ]
            prompt = apply_chat_template(
                [{"role": m["role"], "content": m["content"]} for m in prompt_msgs],
                template=self.template,
            )
            completion = record.output + "\n<eos>"
        else:
            prompt = record.prompt()
            completion = record.completion()
        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)
        # Reserve at least one completion token
        if len(prompt_ids) >= self.max_length:
            # Truncate prompt to make room for at least one completion token
            prompt_ids = prompt_ids[: self.max_length - 1]
            if not completion_ids:
                raise ValueError(f"record {index} has no completion tokens after truncation")
        ids = (prompt_ids + completion_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(ids))
        if prompt_len >= len(ids):
            # No completion fit, skip by truncating prompt further
            if len(prompt_ids) > 0 and len(completion_ids) > 0:
                prompt_ids = prompt_ids[: self.max_length - min(1, len(completion_ids))]
                ids = (prompt_ids + completion_ids)[: self.max_length]
                prompt_len = min(len(prompt_ids), len(ids))
            if prompt_len >= len(ids):
                raise ValueError(
                    f"record {index} prompt consumes full max_length, no completion target"
                )
        # Causal LM: at position i the model predicts ids[i+1]. Supervise the
        # completion, starting from the last prompt token (prompt->completion
        # transition) so the model learns to begin generating the answer.
        labels = [-100] * len(ids)
        start = max(prompt_len - 1, 0)
        if start < len(ids) - 1:
            labels[start:-1] = ids[start + 1 :]
        if all(l == -100 for l in labels):
            raise ValueError(f"record {index} has no supervised completion tokens")
        return {"input_ids": ids, "labels": labels, "record": record}

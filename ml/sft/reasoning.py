"""Reasoning SFT dataset loader (WS-3).

Loads reasoning-heavy instruction records from JSONL with category=reasoning
or a dedicated reasoning mixture. Supports math, logic, multi-hop, and
commonsense task families. Produces InstructionRecord objects consumable by
ml.sft.dataset.SFTDataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ml.instruction.dataset import InstructionRecord, load_jsonl

__all__ = ["ReasoningDataset", "load_reasoning_records"]

# Task families covered by the reasoning stack (mirrors evals/suites/reasoning.yaml).
REASONING_FAMILIES = ("math", "logic", "multihop", "commonsense")


def _family_of(record: InstructionRecord) -> str:
    """Return the reasoning task family for a record, defaulting to 'logic'."""
    raw_family = record.metadata.get("family") if record.metadata else None
    if isinstance(raw_family, str) and raw_family in REASONING_FAMILIES:
        return raw_family
    text = f"{record.instruction} {record.output}".lower()
    if any(k in text for k in ("+", "-", "*", "/", "equals", "=", "number", "digit")):
        return "math"
    if any(k in text for k in ("if ", "then", "therefore", "because", "since", "implies")):
        return "logic"
    if any(k in text for k in ("first", "next", "finally", "step", "common sense", "commonsense")):
        return "commonsense"
    return "logic"


class ReasoningDataset:
    """Filter + annotate reasoning records with their task family."""

    def __init__(self, records: Iterable[InstructionRecord]):
        self.records: list[InstructionRecord] = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> ReasoningDataset:
        raw = load_jsonl(path)
        return cls([r for r in raw if r.category == "reasoning"])

    def by_family(self) -> dict[str, list[InstructionRecord]]:
        grouped: dict[str, list[InstructionRecord]] = {f: [] for f in REASONING_FAMILIES}
        for record in self.records:
            grouped.setdefault(_family_of(record), []).append(record)
        return grouped

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def load_reasoning_records(paths: list[str | Path]) -> list[InstructionRecord]:
    """Load and merge reasoning records from one or more JSONL files."""
    records: list[InstructionRecord] = []
    for p in paths:
        records.extend(ReasoningDataset.from_jsonl(p).records)
    return records
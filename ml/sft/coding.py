"""Coding SFT dataset loader (WS-3).

Loads coding-heavy instruction records from JSONL with category=coding.
Covers function synthesis, bug-fix, explanation, and Bangla-comment code.
Produces InstructionRecord objects consumable by ml.sft.dataset.SFTDataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ml.instruction.dataset import InstructionRecord, load_jsonl

__all__ = ["CodingDataset", "load_coding_records"]

# Task families covered by the coding stack (mirrors evals/suites/coding.yaml).
CODING_FAMILIES = ("function_synthesis", "bug_fix", "explanation", "bangla_comment")


def _family_of(record: InstructionRecord) -> str:
    """Return the coding task family for a record."""
    raw_family = record.metadata.get("family") if record.metadata else None
    if isinstance(raw_family, str) and raw_family in CODING_FAMILIES:
        return raw_family
    text = f"{record.instruction} {record.output}"
    lowered = text.lower()
    if "bangla" in lowered or "বাংলা" in text:
        return "bangla_comment"
    if "bug" in lowered or "fix" in lowered or "error" in lowered or "broken" in lowered:
        return "bug_fix"
    if "explain" in lowered or "explaination" in lowered or "comment" in lowered:
        return "explanation"
    return "function_synthesis"


class CodingDataset:
    """Filter + annotate coding records with their task family."""

    def __init__(self, records: Iterable[InstructionRecord]):
        self.records: list[InstructionRecord] = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> CodingDataset:
        raw = load_jsonl(path)
        return cls([r for r in raw if r.category == "coding"])

    def by_family(self) -> dict[str, list[InstructionRecord]]:
        grouped: dict[str, list[InstructionRecord]] = {f: [] for f in CODING_FAMILIES}
        for record in self.records:
            grouped.setdefault(_family_of(record), []).append(record)
        return grouped

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def load_coding_records(paths: list[str | Path]) -> list[InstructionRecord]:
    """Load and merge coding records from one or more JSONL files."""
    records: list[InstructionRecord] = []
    for p in paths:
        records.extend(CodingDataset.from_jsonl(p).records)
    return records
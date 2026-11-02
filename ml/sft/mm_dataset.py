"""Multimodal instruction dataset loader (WS-4 Phase D).

Loads multimodal instruction records from JSONL. Each record carries text
plus optional modality references (image/audio/document). The loader
produces ``InstructionRecord`` objects with the modality payloads stored in
``metadata`` so the SFT dataset can build the fused input sequence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml.instruction.dataset import InstructionRecord

__all__ = ["MultimodalDataset", "MultimodalRecord", "load_mm_records"]

# Modality keys recognised in the record's ``modalities`` mapping.
MODALITY_KEYS = ("image", "audio", "document")


@dataclass
class MultimodalRecord:
    """An instruction record with optional modality payloads."""

    record: InstructionRecord
    modalities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | InstructionRecord) -> MultimodalRecord:
        """Build a MultimodalRecord from a raw dict or an InstructionRecord.

        Accepting InstructionRecord directly lets MultimodalDataset wrap the
        output of ``ml.instruction.dataset.load_jsonl`` without a
        dict round-trip.
        """
        if isinstance(raw, InstructionRecord):
            return cls(record=raw, modalities={})
        if not isinstance(raw, dict):
            raise TypeError("multimodal record must be an object")
        record = InstructionRecord.from_dict(raw)
        modalities_raw = raw.get("modalities")
        modalities: dict[str, Any] = {}
        if isinstance(modalities_raw, dict):
            for key in MODALITY_KEYS:
                value = modalities_raw.get(key)
                if value is not None:
                    modalities[key] = value
        return cls(record=record, modalities=modalities)

    @property
    def has_modalities(self) -> bool:
        return bool(self.modalities)

    def to_dict(self) -> dict[str, Any]:
        # InstructionRecord is a frozen dataclass; asdict() works on it but
        # returns the raw dataclass fields, so wrap it explicitly.
        return {
            "record": asdict(self.record),
            "modalities": dict(self.modalities),
        }

    @classmethod
    def from_instruction_record(
        cls, record: InstructionRecord, modalities: dict[str, Any] | None = None
    ) -> MultimodalRecord:
        return cls(record=record, modalities=dict(modalities or {}))


class MultimodalDataset:
    """Collection of multimodal instruction records."""

    def __init__(self, records: Iterable[MultimodalRecord]):
        self.records: list[MultimodalRecord] = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> MultimodalDataset:
        """Load multimodal records from JSONL, preserving modality payloads.

        Uses the raw JSON lines (not ``ml.instruction.dataset.load_jsonl``)
        so the ``modalities`` mapping survives the round-trip.
        """
        import json

        records: list[MultimodalRecord] = []
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid multimodal record at line {line_number}: {exc}"
                ) from exc
            records.append(MultimodalRecord.from_dict(raw))
        if not records:
            raise ValueError(f"no multimodal records found in {path}")
        return cls(records)

    def by_modality(self) -> dict[str, list[MultimodalRecord]]:
        grouped: dict[str, list[MultimodalRecord]] = {k: [] for k in MODALITY_KEYS}
        text_only: list[MultimodalRecord] = []
        for record in self.records:
            if not record.has_modalities:
                text_only.append(record)
                continue
            for key in MODALITY_KEYS:
                if key in record.modalities:
                    grouped[key].append(record)
        grouped["text"] = text_only
        return grouped

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def load_mm_records(paths: list[str | Path]) -> list[MultimodalRecord]:
    """Load and merge multimodal records from one or more JSONL files."""
    records: list[MultimodalRecord] = []
    for p in paths:
        records.extend(MultimodalDataset.from_jsonl(p).records)
    return records

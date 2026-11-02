"""Preference dataset — chosen vs rejected pairs (WS-1)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_SOURCES = {"synthetic", "human", "rule-judge"}

@dataclass(frozen=True)
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    source: str = "synthetic"
    task_type: str = "instruction"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreferenceRecord:
        for k in ("prompt", "chosen", "rejected"):
            if not isinstance(raw.get(k), str) or not raw[k].strip():
                raise ValueError(f"{k} must be non-empty string")
        chosen_n = raw["chosen"].strip()
        rejected_n = raw["rejected"].strip()
        # Normalize for comparison (casefold + whitespace collapse)
        import re

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().casefold())

        if _norm(chosen_n) == _norm(rejected_n):
            raise ValueError("chosen and rejected must differ (normalized)")
        source = raw.get("source", "synthetic")
        if source not in _ALLOWED_SOURCES:
            raise ValueError(f"source {source!r} not in {sorted(_ALLOWED_SOURCES)}")
        return cls(raw["prompt"].strip(), chosen_n, rejected_n, source, raw.get("task_type", "instruction"))

def load_preference_jsonl(path: str | Path) -> list[PreferenceRecord]:
    recs = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            recs.append(PreferenceRecord.from_dict(json.loads(line)))
        except Exception as e:
            raise ValueError(f"line {i}: {e}") from e
    if not recs:
        raise ValueError(f"no records in {path}")
    return recs

def split_preference(records: list[PreferenceRecord], val_frac: float = 0.1, seed: int = 0):
    random.Random(seed).shuffle(records)
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]

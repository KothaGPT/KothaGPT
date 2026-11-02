"""WS-4 Phase D: multimodal instruction dataset loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.instruction.dataset import InstructionRecord
from ml.sft.mm_dataset import (
    MODALITY_KEYS,
    MultimodalDataset,
    MultimodalRecord,
    load_mm_records,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    import json

    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_multimodal_record_from_dict_with_modalities(tmp_path: Path):
    raw = {
        "instruction": "বর্ণনা করো",
        "output": "একটি ছবি",
        "language": "bn",
        "category": "instruction",
        "modalities": {"image": "img1.png", "audio": "a.wav"},
    }
    record = MultimodalRecord.from_dict(raw)
    assert isinstance(record.record, InstructionRecord)
    assert record.has_modalities
    assert record.modalities["image"] == "img1.png"
    assert record.modalities["audio"] == "a.wav"


def test_multimodal_record_from_dict_text_only(tmp_path: Path):
    raw = {
        "instruction": "হ্যালো",
        "output": "হ্যালো",
        "language": "bn",
        "category": "instruction",
    }
    record = MultimodalRecord.from_dict(raw)
    assert not record.has_modalities
    assert record.modalities == {}


def test_multimodal_record_rejects_non_object(tmp_path: Path):
    with pytest.raises(TypeError, match="must be an object"):
        MultimodalRecord.from_dict([])  # type: ignore[arg-type]


def test_multimodal_dataset_from_jsonl_groups_by_modality(tmp_path: Path):
    path = tmp_path / "mm.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instruction": "img",
                "output": "desc",
                "language": "en",
                "category": "instruction",
                "modalities": {"image": "a.png"},
            },
            {
                "instruction": "audio",
                "output": "transcript",
                "language": "en",
                "category": "instruction",
                "modalities": {"audio": "a.wav"},
            },
            {
                "instruction": "text",
                "output": "answer",
                "language": "en",
                "category": "instruction",
            },
        ],
    )
    dataset = MultimodalDataset.from_jsonl(path)
    grouped = dataset.by_modality()
    assert len(grouped["image"]) == 1
    assert len(grouped["audio"]) == 1
    assert len(grouped["document"]) == 0
    assert len(grouped["text"]) == 1
    assert len(dataset) == 3


def test_load_mm_records_merges_multiple_files(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [{"instruction": "x", "output": "y", "language": "en", "category": "instruction"}])
    _write_jsonl(b, [{"instruction": "p", "output": "q", "language": "en", "category": "instruction"}])
    records = load_mm_records([a, b])
    assert len(records) == 2


def test_multimodal_record_to_dict_round_trips():
    raw = {
        "instruction": "x",
        "output": "y",
        "language": "en",
        "category": "instruction",
        "modalities": {"image": "a.png"},
    }
    record = MultimodalRecord.from_dict(raw)
    payload = record.to_dict()
    assert payload["modalities"] == {"image": "a.png"}
    assert payload["record"]["instruction"] == "x"


def test_modality_keys_cover_all_supported_modalities():
    assert set(MODALITY_KEYS) == {"image", "audio", "document"}

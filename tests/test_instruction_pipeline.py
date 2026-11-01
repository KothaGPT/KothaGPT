from __future__ import annotations

import gzip
import json

import pytest

from data.manifest import DatasetRegistry, DatasetSource
from data.pipeline import scoring
from data.pipeline.cli import _build_parser
from data.pipeline.instruction import run_instruction_pipeline
from data.pipeline.schema import normalize_instruction_record


def _record(index: int) -> dict:
    return {
        "instruction": f"বাংলা ভাষার একটি গুরুত্বপূর্ণ বিষয় ব্যাখ্যা করুন, উদাহরণ {index}।",
        "output": f"বাংলা ভাষা সমৃদ্ধ ও বৈচিত্র্যময়। বিষয় {index} নিয়ে বিস্তারিত আলোচনা করা যায়।",
        "source": "curated-bangla",
        "license": "cc0-1.0",
    }


def test_schema_canonicalizes_alpaca_and_validates_turns():
    record = normalize_instruction_record(_record(1))
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert record["messages"][-1]["content"].startswith("বাংলা ভাষা")

    with pytest.raises(ValueError, match="end with"):
        normalize_instruction_record(
            {
                "messages": [
                    {"role": "user", "content": "প্রশ্ন"},
                    {"role": "assistant", "content": "উত্তর"},
                    {"role": "user", "content": "আরেকটি প্রশ্ন"},
                ],
                "source": "curated",
                "license": "cc0-1.0",
            }
        )


def test_quality_score_is_bounded_and_explained():
    record = normalize_instruction_record(_record(2))
    result = scoring.score_instruction_record(record)
    assert 0 <= result["quality_score"] <= 1
    assert set(result["quality_components"]) == {
        "language",
        "diversity",
        "length",
        "provenance",
    }


def test_instruction_pipeline_gates_scores_splits_and_versions(tmp_path):
    source = tmp_path / "curated.jsonl"
    records = [_record(index) for index in range(12)]
    records.append(
        {
            "messages": [
                {"role": "user", "content": "আমার ইমেইল কী?"},
                {"role": "assistant", "content": "আমার ঠিকানা user@example.com"},
            ],
            "source": "curated-bangla",
            "license": "cc0-1.0",
        }
    )
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    out_root = tmp_path / "processed"
    summary = run_instruction_pipeline(
        source,
        out_root,
        min_score=0.5,
        validation_ratio=0.2,
        test_ratio=0.2,
        strict=False,
    )

    assert summary["raw"] == 13
    assert summary["after_safety"] == 12
    assert summary["quality_scores"]["count"] == 12
    assert sum(summary["quality_scores"]["histogram"].values()) == 12
    assert summary["train"] + summary["validation"] + summary["test"] == 12
    version_dir = out_root / summary["version_id"]
    manifest = json.loads((version_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["quality_scores"]["count"] == 12
    assert any(shard["split"] == "test" for shard in manifest["shards"])
    report = (version_dir / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "Quality score distribution" in report

    shard = next((version_dir / "instruction" / "train").glob("*.gz"), None)
    if shard is None:
        shard = next((version_dir / "instruction" / "validation").glob("*.gz"))
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        output_record = json.loads(handle.readline())
    assert "text" not in output_record
    assert output_record["messages"][-1]["role"] == "assistant"
    assert 0.5 <= output_record["quality_score"] <= 1
    assert manifest["config"]["input_sha256"]


def test_instruction_cli_exposes_gates_and_score_threshold():
    args = _build_parser().parse_args(
        ["instruction", "run", "curated.jsonl", "--min-score", "0.7", "--dev-bypass"]
    )
    assert args.min_score == 0.7
    assert args.dev_bypass


def test_instruction_pipeline_requires_registered_training_source(tmp_path):
    source = tmp_path / "curated.jsonl"
    source.write_text(json.dumps(_record(1), ensure_ascii=False) + "\n", encoding="utf-8")
    registry = DatasetRegistry()
    registry.register(
        DatasetSource(
            name="curated-bangla",
            url="https://example.org/curated",
            license="cc0-1.0",
            provenance="curated by the project",
        )
    )
    summary = run_instruction_pipeline(
        source,
        tmp_path / "registered-output",
        validation_ratio=0,
        test_ratio=0,
        registry=registry,
    )
    assert summary["after_provenance"] == 1
    assert summary["train"] == 1

    with pytest.raises(ValueError, match="provenance=1"):
        run_instruction_pipeline(
            source,
            tmp_path / "unregistered-output",
            validation_ratio=0,
            test_ratio=0,
            registry=DatasetRegistry(),
        )


def test_instruction_pipeline_masks_pii_when_requested(tmp_path):
    source = tmp_path / "pii.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "আমার ইমেইল user@example.com"},
            {"role": "assistant", "content": "আমি সাহায্য করতে পারি।"},
        ],
        "source": "curated-bangla",
        "license": "cc0-1.0",
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = run_instruction_pipeline(
        source,
        tmp_path / "masked-output",
        validation_ratio=0,
        test_ratio=0,
        pii_mode="mask",
        strict=False,
    )
    shard = next(
        (tmp_path / "masked-output" / summary["version_id"] / "instruction" / "train").glob(
            "*.gz"
        )
    )
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        output_record = json.loads(handle.readline())
    assert "user@example.com" not in json.dumps(output_record, ensure_ascii=False)
    assert "[PII]" in output_record["messages"][0]["content"]

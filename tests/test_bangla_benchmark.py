"""Tests for the WS-8 Bangla benchmark dataset (data/benchmarks/bangla/v1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.benchmarks.bangla.generate import (
    build_generation,
    build_qa,
    build_summarization,
    build_translation,
    generate,
)
from data.benchmarks.bangla.robustness import build_robustness

V1 = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"

MIN_COUNTS = {
    "bangla_qa": 500,
    "bangla_translation": 1000,
    "bangla_summarization": 100,
    "bangla_generation": 20,
    "bangla_robustness": 20,
}


@pytest.fixture(scope="module")
def dataset() -> dict[str, list[dict]]:
    return {
        "bangla_qa": build_qa(),
        "bangla_translation": build_translation(),
        "bangla_summarization": build_summarization(),
        "bangla_generation": build_generation(),
        "bangla_robustness": build_robustness(),
    }


@pytest.mark.parametrize("task,minimum", MIN_COUNTS.items())
def test_task_meets_plan_minimum(dataset, task, minimum):
    assert len(dataset[task]) >= minimum


@pytest.mark.parametrize("task", MIN_COUNTS)
def test_generated_matches_committed_v1(task, dataset):
    if not (V1 / f"{task}.jsonl").exists():
        pytest.skip("v1 JSONL not committed in this checkout")
    committed = [
        json.loads(line) for line in (V1 / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(committed) == len(dataset[task])
    assert [r["record_id"] for r in committed] == [r["record_id"] for r in dataset[task]]


def test_qa_answers_are_verbatim_spans(dataset):
    for record in dataset["bangla_qa"]:
        assert record["reference"] in record["passage"]


def test_translation_has_both_directions(dataset):
    langs = {(r["source_lang"], r["target_lang"]) for r in dataset["bangla_translation"]}
    assert ("bn", "en") in langs
    assert ("en", "bn") in langs


def test_all_records_have_stable_split_and_id(dataset):
    for task, records in dataset.items():
        for record in records:
            assert record["split"] in ("dev", "test")
            assert record["record_id"].startswith(
                {
                    "bangla_qa": "qa:",
                    "bangla_translation": "trans:",
                    "bangla_summarization": "sum:",
                    "bangla_generation": "gen:",
                    "bangla_robustness": "robustness:",
                }[task]
            )


def test_generation_has_reference(dataset):
    for record in dataset["bangla_generation"]:
        assert record["prompt"]
        assert record["reference"]
        assert len(record["reference"].split()) > 5


def test_robustness_records_are_committed_and_have_stable_splits():
    records = build_robustness()
    committed = [
        json.loads(line)
        for line in (V1 / "bangla_robustness.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert committed == records
    assert len(records) >= 20
    assert len({record["record_id"] for record in records}) == len(records)
    for record in records:
        assert record["split"] in ("dev", "test")
        assert record["dialect"] in {"standard", "colloquial"}
        assert record["region"] == "bn-BD"
        assert record["script_pattern"]
        assert record["phenomena"]
        assert record["domain"]


def test_robustness_taxonomy_covers_required_input_types():
    records = build_robustness()
    assert {record["dialect"] for record in records} == {"standard", "colloquial"}
    assert {record["family"] for record in records} >= {
        "formal",
        "colloquial",
        "code_switched",
        "transliterated",
        "noisy_mixed",
    }
    assert {record["script_pattern"] for record in records} >= {
        "bn",
        "bn_en",
        "latin_bn",
        "bn_noisy",
        "bn_en_noisy",
        "latin_bn_noisy",
    }
    noisy = [record for record in records if record["family"] == "noisy_mixed"]
    assert len(noisy) >= 4
    noise_tags = {tag for record in noisy for tag in record["phenomena"]}
    assert {"repeated_punctuation", "spacing_noise", "repeated_characters"} <= noise_tags


def test_core_generator_preserves_robustness_manifest_entry(tmp_path):
    generate(tmp_path)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    counts = manifest["tasks"]["bangla_robustness"]
    assert counts["total"] == len(build_robustness())
    assert counts["dev"] + counts["test"] == counts["total"]

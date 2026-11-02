"""Tests for the WS-9 Bangla eval harness (evals/metrics.py, evals/run.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.metrics import (
    bengali_script_ratio,
    context_token_coverage,
    context_unsupported_token_rate,
    exact_match,
    language_detection,
    mean_ci,
    normalize_answer,
    rouge,
)
from evals.run import MockTarget, PredictionsTarget, main, run_suite

SUITE = Path(__file__).parent.parent / "evals/suites/bangla.yaml"
DATA = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"
ROBUSTNESS_SUITE = Path(__file__).parent.parent / "evals/suites/bangla-robustness.yaml"
HALLUCINATION_SUITE = Path(__file__).parent.parent / "evals/suites/hallucination.yaml"
HALLUCINATION_DATA = Path(__file__).parent.parent / "data/benchmarks/hallucination/v1"


class UnsupportedClaimTarget:
    def generate(self, prompt, record):
        if record["record_id"] == "hallucination:bn:unsupported":
            return "ঢাকা চাঁদে"
        return record["reference"]


def test_exact_match_ignores_punctuation_and_case():
    assert exact_match("ঢাকা।", "ঢাকা") == 1.0
    assert exact_match("Hello!", "hello") == 1.0
    assert exact_match("বাংলা", "ইংরেজি") == 0.0


def test_context_token_coverage_is_a_lexical_support_baseline():
    assert context_token_coverage("ঢাকা বাংলাদেশের রাজধানী।", "বাংলাদেশের রাজধানী ঢাকা") == 1.0
    assert context_token_coverage("ঢাকা চাঁদে", "ঢাকা") == 0.5
    assert context_token_coverage("", "ঢাকা") == 0.0
    assert context_unsupported_token_rate("ঢাকা বাংলাদেশের রাজধানী।", "বাংলাদেশের রাজধানী ঢাকা") == 0.0
    assert context_unsupported_token_rate("ঢাকা চাঁদে", "ঢাকা") == 0.5
    assert context_unsupported_token_rate("", "ঢাকা") == 0.0


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer("  বাংলা    ভাষা  ") == "বাংলা ভাষা"


def test_language_detection():
    assert language_detection("বাংলা ভাষা") == "bn"
    assert language_detection("Hello world") == "en"
    assert language_detection("Hello বাংলা") == "bn"


def test_bengali_script_ratio():
    assert bengali_script_ratio("বাংলা") == 1.0
    assert bengali_script_ratio("Bangla") == 0.0
    assert bengali_script_ratio("বাংলা ভাষা") == 1.0


def test_rouge_perfect_match_is_one():
    scores = rouge("বাংলা ভাষা বাংলাদেশের ভাষা", "বাংলা ভাষা বাংলাদেশের ভাষা")
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_disjoint_is_zero():
    scores = rouge("এক দুই তিন", "চার পাঁচ ছয়")
    assert scores["rouge1"] == 0.0


def test_rouge_partial_overlap():
    scores = rouge("বাংলা ভাষা সুন্দর", "বাংলা ভাষা")
    assert scores["rouge1"] > 0.0
    assert scores["rougeL"] > 0.0


def test_mean_ci_bounds():
    stats = mean_ci([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_mean_ci_empty():
    assert mean_ci([])["mean"] == 0.0


@pytest.mark.parametrize(
    "task",
    ["bangla_qa", "bangla_translation", "bangla_summarization", "bangla_generation"],
)
def test_suite_runs_all_tasks_with_mock_target(task):
    results = run_suite(SUITE, data_dir=DATA, split="all", target=MockTarget())
    summary = results["summary"][task]
    assert summary["instances"] > 0
    assert any(
        key in summary["metrics"] for key in ("exact_match", "rouge1", "bengali_script_ratio")
    )


def test_mock_target_returns_gold_references():
    record = {"reference": "বাংলা ভাষা"}
    assert MockTarget().generate("প্রশ্ন?", record) == "বাংলা ভাষা"


def test_qa_exact_match_is_perfect_under_mock():
    results = run_suite(SUITE, data_dir=DATA, split="dev", target=MockTarget())
    em = results["summary"]["bangla_qa"]["metrics"]["exact_match"]
    assert em["mean"] == 1.0


def test_translation_rouge_is_perfect_under_mock():
    results = run_suite(SUITE, data_dir=DATA, split="dev", target=MockTarget())
    r1 = results["summary"]["bangla_translation"]["metrics"]["rouge1"]
    assert r1["mean"] == 1.0


def test_robustness_suite_reports_metrics_by_input_family():
    results = run_suite(ROBUSTNESS_SUITE, split="all", target=MockTarget())
    summary = results["summary"]["bangla_robustness"]
    assert summary["instances"] == 24
    assert summary["metrics"]["rouge1"]["mean"] == 1.0
    assert summary["metrics"]["language_correct"]["mean"] == 1.0
    assert set(results["families"]["bangla_robustness"]) == {
        "formal",
        "colloquial",
        "code_switched",
        "transliterated",
        "noisy_mixed",
    }


def test_hallucination_suite_reports_context_token_coverage():
    results = run_suite(
        HALLUCINATION_SUITE,
        data_dir=HALLUCINATION_DATA,
        split="all",
        target=UnsupportedClaimTarget(),
    )
    coverage = results["summary"]["grounded_qa"]["metrics"]["context_token_coverage"]
    assert coverage["mean"] == 0.75
    unsupported = results["summary"]["grounded_qa"]["metrics"]["lexical_unsupported_token_rate"]
    assert unsupported["mean"] == 0.25
    families = results["families"]["grounded_qa"]
    assert families["bn_factual"]["metrics"]["lexical_unsupported_token_rate"]["mean"] == 0.5
    assert families["en_factual"]["metrics"]["lexical_unsupported_token_rate"]["mean"] == 0.0


def test_predictions_target_matches_records_by_id(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"record_id":"hallucination:en:supported","prediction":"Dhaka is the capital of Bangladesh."}\n'
        '{"record_id":"hallucination:bn:unsupported","prediction":"ঢাকা চাঁদে"}\n',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        '{"record_id":"hallucination:bn:unsupported","prediction":"ঢাকা বাংলাদেশের রাজধানী"}\n'
        '{"record_id":"hallucination:en:supported","prediction":"Dhaka is the capital of Bangladesh."}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"
    assert main(
        [
            "--suite",
            str(HALLUCINATION_SUITE),
            "--data-dir",
            str(HALLUCINATION_DATA),
            "--split",
            "all",
            "--target",
            "predictions",
            "--predictions",
            str(predictions),
            "--compare-predictions",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--name",
            "hallucination",
        ]
    ) == 0
    result_path = next(out_dir.glob("*.json"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert results["target"] == "predictions"
    assert results["summary"]["grounded_qa"]["metrics"]["lexical_unsupported_token_rate"]["mean"] == 0.25
    delta = results["comparison"]["tasks"]["grounded_qa"]["metrics"]["lexical_unsupported_token_rate"]["delta"]
    assert delta == 0.25
    family_delta = results["comparison"]["families"]["grounded_qa"]["bn_factual"]["metrics"]["lexical_unsupported_token_rate"]["delta"]
    assert family_delta == 0.5
    report = result_path.with_suffix(".REPORT.md").read_text(encoding="utf-8")
    assert "Results by family" in report
    assert "bn_factual" in report and "en_factual" in report
    assert "Comparison against baseline" in report


def test_predictions_target_rejects_missing_record(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"record_id":"hallucination:en:supported","prediction":"supported"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hallucination:bn:unsupported"):
        run_suite(
            HALLUCINATION_SUITE,
            data_dir=HALLUCINATION_DATA,
            split="all",
            target=PredictionsTarget(predictions),
        )

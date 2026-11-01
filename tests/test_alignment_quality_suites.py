"""Tests for WS-7 helpfulness and WS-8 Bangla quality evals."""

from pathlib import Path

import pytest

from evals.metrics import bangla_quality_metrics, helpfulness_metrics
from evals.run import run_suite

_ROOT = Path(__file__).parent.parent


def test_helpfulness_metrics_reward_supported_concise_answers():
    metrics = helpfulness_metrics(
        "ঢাকা বাংলাদেশের রাজধানী।",
        "ঢাকা বাংলাদেশের রাজধানী।",
    )
    assert metrics["answerability"] == 1.0
    assert metrics["information_sufficiency"] == 1.0
    assert metrics["verbosity_fit"] == 1.0
    assert metrics["non_refusal"] == 1.0
    assert metrics["helpfulness_score"] == 1.0


def test_helpfulness_metrics_penalize_non_answers_and_refusals():
    unknown = helpfulness_metrics("আমি জানি না।", "বাংলাদেশের রাজধানী ঢাকা।", 4)
    refused = helpfulness_metrics(
        "I cannot help with that request.",
        "The capital is Dhaka.",
    )
    assert unknown["information_sufficiency"] == 0.0
    assert unknown["helpfulness_score"] < 0.75
    assert refused["non_refusal"] == 0.0
    assert refused["helpfulness_score"] < 0.75


def test_bangla_quality_metrics_detect_bengali_and_transliteration():
    good = bangla_quality_metrics("ঢাকা বাংলাদেশের রাজধানী।", "ঢাকা বাংলাদেশের রাজধানী।")
    transliterated = bangla_quality_metrics("Dhaka Bangladesh er rajdhani")
    assert good["bengali_script_ratio"] == 1.0
    assert good["language_correct"] == 1.0
    assert good["bangla_quality_score"] == 1.0
    assert transliterated["bengali_script_ratio"] == 0.0
    assert transliterated["language_correct"] == 0.0


@pytest.mark.parametrize(
    ("suite_name", "categories", "required_metrics"),
    [
        (
            "helpfulness",
            ("helpful", "unhelpful"),
            ("helpfulness_score", "information_sufficiency", "verbosity_fit", "non_refusal"),
        ),
        (
            "bangla-quality",
            ("good", "poor"),
            ("bangla_quality_score", "bengali_script_ratio", "language_correct"),
        ),
    ],
)
def test_quality_suite_runs_and_reports_metrics(suite_name, categories, required_metrics):
    results = run_suite(_ROOT / f"evals/suites/{suite_name}.yaml")

    assert results["suite"] == suite_name
    expected_cases = 5 if suite_name == "helpfulness" else 4
    assert len(results["cases"]) == expected_cases
    assert all(case["passed"] for case in results["cases"])
    for category in categories:
        summary = results["summary"][category]
        assert summary["metrics"]["pass_rate"]["mean"] == 1.0
        assert set(required_metrics) <= summary["metrics"].keys()

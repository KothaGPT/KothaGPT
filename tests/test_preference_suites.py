"""WS-4/WS-5/WS-6/WS-9 — preference, hallucination, safety, refusal suites."""

from pathlib import Path

import pytest

from evals.metrics import (
    preference_judge,
    refusal_classifier,
    semantic_contradiction,
    semantic_entailment,
)
from evals.run import MockTarget, run_suite

_ROOT = Path(__file__).parent.parent


def test_preference_judge_picks_chosen():
    result = preference_judge(
        "ঢাকা বাংলাদেশের রাজধানী।",
        "ঢাকা ভারতের রাজধানী।",
        "ঢাকা বাংলাদেশের রাজধানী",
    )
    assert result["win"] == 1.0
    assert result["tie"] == 0.0
    assert result["chosen_score"] > result["rejected_score"]


def test_preference_judge_tie_when_equal_overlap():
    result = preference_judge("hello world", "hello there", "hello world")
    # chosen overlaps fully with reference; rejected overlaps partially.
    assert result["chosen_score"] >= result["rejected_score"]


def test_semantic_entailment_supported_answer():
    assert semantic_entailment("ঢাকা বাংলাদেশের রাজধানী", "ঢাকা বাংলাদেশের রাজধানী।") > 0.5


def test_semantic_contradiction_high_for_unsupported():
    score = semantic_contradiction(" ঢাকা ভারতের রাজধানী", "ঢাকা বাংলাদেশের রাজধানী।")
    assert score >= 0.5


def test_refusal_classifier_three_way():
    assert refusal_classifier("I cannot help with that.") == "refuse"
    assert refusal_classifier("The capital is Dhaka.") == "answer"
    assert (
        refusal_classifier("I can't help with that, but here's a map of Dhaka.")
        == "partial"
    )
    assert refusal_classifier("আমি এটা করতে পারব না, কিন্তু এটার সমাধান হলো যে ঢাকা বাংলাদেশের রাজধানী।") == "partial"
    assert refusal_classifier("দুঃখিত, আমি এটা করতে পারব না।") == "refuse"


@pytest.mark.parametrize(
    "suite",
    ["preference", "hallucination", "safety", "refusal"],
)
def test_suite_runs_clean(suite):
    data_dir = (
        _ROOT / "data/benchmarks/hallucination/v1" if suite == "hallucination" else None
    )
    results = run_suite(
        _ROOT / f"evals/suites/{suite}.yaml",
        data_dir=data_dir,
        target=MockTarget(),
    )
    assert results["suite"]
    # Every case must pass when the target returns the expected outcome.
    for case in results["cases"]:
        assert case["passed"], (suite, case["id"], case["observed"], case["expect"])


def test_preference_suite_win_rate_is_1():
    results = run_suite(
        _ROOT / "evals/suites/preference.yaml",
        target=MockTarget(),
    )
    win = results["summary"]["preference"]["metrics"]["win"]["mean"]
    assert win == 1.0


def test_safety_suite_harmful_all_refused():
    results = run_suite(
        _ROOT / "evals/suites/safety.yaml",
        target=MockTarget(),
    )
    harmful = results["summary"]["harmful"]
    assert harmful["passed"] == harmful["instances"]
    benign = results["summary"]["benign"]
    assert benign["passed"] == benign["instances"]


def test_refusal_suite_partial_classified():
    results = run_suite(
        _ROOT / "evals/suites/refusal.yaml",
        target=MockTarget(),
    )
    partial = results["summary"]["partial"]
    assert partial["passed"] == partial["instances"]


def test_hallucination_suite_reports_semantic_metrics():
    results = run_suite(
        _ROOT / "evals/suites/hallucination.yaml",
        data_dir=_ROOT / "data/benchmarks/hallucination/v1",
        target=MockTarget(),
    )
    metrics = results["summary"]["grounded_qa"]["metrics"]
    assert "semantic_entailment" in metrics
    assert "semantic_contradiction" in metrics


def test_preference_suite_tasks_scored():
    results = run_suite(
        _ROOT / "evals/suites/preference.yaml",
        target=MockTarget(),
    )
    assert results["summary"]["preference"]["instances"] == 7
    assert results["summary"]["preference"]["metrics"]["win"]["mean"] == 1.0
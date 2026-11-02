"""WS-6: Tier-aware regression gate tests."""

from __future__ import annotations

from typing import Any

from evals.regression import tier_critical_metrics, tier_regression_gate


def _result(summary: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    return summary


def _group(metrics: dict[str, float]) -> dict[str, Any]:
    return {"metrics": {k: {"mean": v} for k, v in metrics.items()}}


def test_tier_critical_metrics_returns_per_tier_map():
    nano = tier_critical_metrics("nano")
    large = tier_critical_metrics("large")
    assert nano != large
    # Large should have at least as many critical metrics as nano.
    assert len(large) >= len(nano)


def test_tier_critical_metrics_unknown_tier_raises():
    try:
        tier_critical_metrics("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown tier")


def test_nano_lenient_on_reasoning():
    nano = tier_critical_metrics("nano")
    # Nano does not include preference.win (reasoning-heavy).
    assert "preference.win" not in nano
    # Nano still enforces safety.
    assert ("safety.harmful", "pass_rate", 0.0) in nano.values()


def test_large_strict_on_everything():
    large = tier_critical_metrics("large")
    # Large enforces safety, refusal, preference, hallucination.
    assert ("safety.harmful", "pass_rate", 0.0) in large.values()
    assert ("safety.benign", "pass_rate", 0.0) in large.values()
    assert ("preference", "win", 0.0) in large.values()
    assert ("grounded_qa", "semantic_contradiction", -0.05) in large.values()


def test_tier_regression_gate_passes_when_no_regression():
    tier = "large"
    baseline_metrics = {
        "safety.harmful": {"pass_rate": {"mean": 1.0}},
        "safety.benign": {"pass_rate": {"mean": 1.0}},
        "refusal.harmful": {"pass_rate": {"mean": 1.0}},
        "refusal.benign": {"pass_rate": {"mean": 1.0}},
        "refusal.partial": {"pass_rate": {"mean": 1.0}},
        "preference": {"win": {"mean": 0.6}},
        "grounded_qa": {
            "lexical_unsupported_token_rate": {"mean": 0.01},
            "semantic_entailment": {"mean": 0.9},
            "semantic_contradiction": {"mean": 0.01},
        },
    }
    candidate_metrics = dict(baseline_metrics)
    decision = tier_regression_gate(
        tier,
        {"summary": {k: _group(v) for k, v in candidate_metrics.items()}},
        {"summary": {k: _group(v) for k, v in baseline_metrics.items()}},
    )
    assert decision["passed"] is True
    assert decision["tier"] == "large"
    assert decision["failures"] == []


def test_tier_regression_gate_fails_on_safety_regression():
    tier = "large"
    baseline = {
        "safety.harmful": {"pass_rate": {"mean": 1.0}},
    }
    candidate = {
        "safety.harmful": {"pass_rate": {"mean": 0.5}},  # regression below 0.0 delta
    }
    decision = tier_regression_gate(
        tier,
        _result({"summary": {k: _group(v) for k, v in candidate.items()}}),
        _result({"summary": {k: _group(v) for k, v in baseline.items()}}),
    )
    assert decision["passed"] is False
    assert any("safety.harmful" in f for f in decision["failures"])


def test_tier_regression_gate_fails_on_hallucination_regression():
    tier = "large"
    baseline = {
        "grounded_qa": {
            "lexical_unsupported_token_rate": {"mean": 0.01},
        },
    }
    candidate = {
        "grounded_qa": {
            "lexical_unsupported_token_rate": {"mean": 0.2},  # regression > 0.05
        },
    }
    decision = tier_regression_gate(
        tier,
        _result({"summary": {k: _group(v) for k, v in candidate.items()}}),
        _result({"summary": {k: _group(v) for k, v in baseline.items()}}),
    )
    assert decision["passed"] is False
    assert any("hallucination" in f for f in decision["failures"])


def test_nano_passes_without_preference_metric():
    # Nano does not require preference.win, so missing it should not fail.
    tier = "nano"
    baseline = {
        "safety.harmful": {"pass_rate": {"mean": 1.0}},
        "refusal.harmful": {"pass_rate": {"mean": 1.0}},
        "grounded_qa": {"lexical_unsupported_token_rate": {"mean": 0.01}},
    }
    candidate = dict(baseline)
    decision = tier_regression_gate(
        tier,
        _result({"summary": {k: _group(v) for k, v in candidate.items()}}),
        _result({"summary": {k: _group(v) for k, v in baseline.items()}}),
    )
    assert decision["passed"] is True

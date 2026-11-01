"""Tests for the release-grade benchmark gate (KGPT-004)."""

from __future__ import annotations

import json
from pathlib import Path

from evals.gate import _get_metric, evaluate_gate, load_result, load_results


def _summary(group: str, metric: str, value: float) -> dict:
    return {
        group: {
            "instances": 1,
            "metrics": {metric: {"mean": value, "std": 0.0, "ci_low": value, "ci_high": value}},
        }
    }


def _full_result(group: str, metric: str, value: float) -> dict:
    return {"summary": _summary(group, metric, value)}


def test_get_metric_returns_none_for_missing_group():
    assert _get_metric({}, "bangla_qa", "exact_match") is None


def test_get_metric_returns_none_for_missing_metric():
    assert _get_metric({"bangla_qa": {"metrics": {}}}, "bangla_qa", "exact_match") is None


def test_get_metric_returns_mean_for_present_metric():
    summary = _summary("bangla_qa", "exact_match", 0.9)
    assert _get_metric(summary, "bangla_qa", "exact_match") == 0.9


def test_gate_passes_when_candidate_meets_baseline():
    candidate = _full_result("bangla_qa", "exact_match", 0.9)
    baseline = _full_result("bangla_qa", "exact_match", 0.8)
    decision = evaluate_gate(candidate, baseline, critical={}, warn={})
    assert decision["passed"] is True
    assert decision["failures"] == []


def test_gate_fails_on_critical_regression():
    candidate = _full_result("bangla_qa", "exact_match", 0.7)
    baseline = _full_result("bangla_qa", "exact_match", 0.9)
    critical = {"bangla_qa.exact_match": ("bangla_qa", "exact_match", 0.0)}
    decision = evaluate_gate(candidate, baseline, critical=critical)
    assert decision["passed"] is False
    assert any("regression" in f for f in decision["failures"])


def test_gate_fails_when_metric_missing_in_baseline():
    candidate = _full_result("bangla_qa", "exact_match", 0.9)
    baseline: dict = {"summary": {}}
    critical = {"bangla_qa.exact_match": ("bangla_qa", "exact_match", 0.0)}
    decision = evaluate_gate(candidate, baseline, critical=critical)
    assert decision["passed"] is False
    assert any("missing metric" in f for f in decision["failures"])


def test_gate_warns_but_does_not_fail_on_warn_threshold():
    candidate = _full_result("bangla_qa", "exact_match", 0.5)
    baseline = _full_result("bangla_qa", "exact_match", 0.9)
    critical = {"bangla_qa.exact_match": ("bangla_qa", "exact_match", 0.0)}
    decision = evaluate_gate(candidate, baseline, critical=critical)
    # exact_match is critical, so it fails; warn metrics are informational only.
    assert decision["passed"] is False


def test_gate_supports_custom_critical_and_warn_metrics():
    candidate = _full_result("custom", "score", 0.5)
    baseline = _full_result("custom", "score", 0.9)
    critical = {"custom.score": ("custom", "score", -0.1)}
    decision = evaluate_gate(candidate, baseline, critical=critical)
    assert decision["passed"] is False


def test_load_result_reads_json_file(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"suite": "bangla", "summary": {}}), encoding="utf-8")
    data = load_result(path)
    assert data["suite"] == "bangla"


def test_load_result_rejects_non_object(tmp_path: Path):
    import pytest

    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TypeError):
        load_result(path)


def test_load_results_combines_summary_groups(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_full_result("bangla_qa", "exact_match", 0.9)))
    second.write_text(json.dumps(_full_result("safety", "pass_rate", 1.0)))

    combined = load_results([first, second])

    assert _get_metric(combined["summary"], "bangla_qa", "exact_match") == 0.9
    assert _get_metric(combined["summary"], "safety", "pass_rate") == 1.0


def test_load_results_namespaces_case_suite_categories(tmp_path: Path):
    safety = tmp_path / "safety.json"
    refusal = tmp_path / "refusal.json"
    safety.write_text(
        json.dumps(
            {
                "suite": "safety",
                "summary": _summary("benign", "pass_rate", 1.0),
                "cases_by_category": {"benign": []},
            }
        )
    )
    refusal.write_text(
        json.dumps(
            {
                "suite": "refusal",
                "summary": _summary("benign", "pass_rate", 1.0),
                "cases_by_category": {"benign": []},
            }
        )
    )

    combined = load_results([safety, refusal])

    assert _get_metric(combined["summary"], "safety.benign", "pass_rate") == 1.0
    assert _get_metric(combined["summary"], "refusal.benign", "pass_rate") == 1.0


def test_load_results_rejects_duplicate_summary_groups(tmp_path: Path):
    import pytest

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_full_result("safety", "pass_rate", 1.0)))
    second.write_text(json.dumps(_full_result("safety", "pass_rate", 1.0)))

    with pytest.raises(ValueError, match="duplicate summary group 'safety'"):
        load_results([first, second])


def test_gate_details_record_delta():
    candidate = _full_result("bangla_qa", "exact_match", 0.95)
    baseline = _full_result("bangla_qa", "exact_match", 0.90)
    decision = evaluate_gate(candidate, baseline)
    detail = decision["details"]["bangla_qa.exact_match"]
    assert detail["candidate"] == 0.95
    assert detail["baseline"] == 0.90
    assert abs(detail["delta"] - 0.05) < 1e-9
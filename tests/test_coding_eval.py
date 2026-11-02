"""WS-3 coding stack eval: execution-based pass@k and syntax-validity scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.coding_eval import (
    evaluate_coding_records,
    score_prediction,
    syntax_valid,
    _extract_code,
)


def test_extract_code_strips_markdown_fences():
    prediction = "```python\ndef add(a, b):\n    return a + b\n```"
    code = _extract_code(prediction)
    assert code.startswith("def add")
    assert "```" not in code


def test_extract_code_keeps_plain_code():
    code = _extract_code("def square(n): return n * n")
    assert code == "def square(n): return n * n"


def test_syntax_valid_accepts_parseable_code():
    valid, error = syntax_valid("def f():\n    return 1")
    assert valid is True
    assert error is None


def test_syntax_valid_rejects_invalid_code():
    valid, error = syntax_valid("def f(:\n    return 1")
    assert valid is False
    assert error is not None
    assert "SyntaxError" in error


def test_score_prediction_text_match():
    record = {"output": "def add(a, b):\n    return a + b"}
    scores = score_prediction("def add(a, b):\n    return a + b", record)
    assert scores["exact_match"] == 1.0
    assert scores["normalized_match"] == 1.0
    assert scores["syntax_valid"] == 1.0


def test_score_prediction_normalized_match():
    record = {"output": "def add(a, b):\n    return a + b"}
    scores = score_prediction("def add(a, b):\n    return a + b  # comment", record)
    assert scores["exact_match"] == 0.0
    assert scores["normalized_match"] == 0.0
    assert scores["syntax_valid"] == 1.0


def test_score_prediction_syntax_invalid():
    record = {"output": "def add(a, b):\n    return a + b"}
    scores = score_prediction("def add(:\n    return a + b", record)
    assert scores["syntax_valid"] == 0.0


def test_score_prediction_execution_pass(tmp_path: Path):
    # The exec block is a test harness that assumes the prediction's function
    # is already defined; the scorer prepends the prediction before running.
    record = {
        "output": "def f(): return 4",
        "exec": {
            "code": "print(f())",
            "expected_stdout": "4",
        },
    }
    scores = score_prediction("def f():\n    return 4", record)
    assert scores["execution_pass"] == 1.0


def test_score_prediction_execution_fail(tmp_path: Path):
    # The prediction is executed, not the gold reference. A wrong answer
    # (f() returns 5 when the expected stdout is "4") must score 0.
    record = {
        "output": "def f(): return 4",
        "exec": {
            "code": "print(f())",
            "expected_stdout": "4",
        },
    }
    scores = score_prediction("def f():\n    return 5", record)
    assert scores["execution_pass"] == 0.0


def test_score_prediction_execution_pass_uses_prediction_not_gold(tmp_path: Path):
    # The gold output is "def f(): return 5" but the model predicts
    # "def f(): return 4" — execution_pass must be 1.0 (harness prints 4)
    # even though exact_match is 0.
    record = {
        "output": "def f():\n    return 5",
        "exec": {
            "code": "print(f())",
            "expected_stdout": "4",
        },
    }
    scores = score_prediction("def f():\n    return 4", record)
    assert scores["exact_match"] == 0.0
    assert scores["execution_pass"] == 1.0


def test_score_prediction_execution_not_run_when_no_exec_block():
    record = {"output": "def f(): pass"}
    scores = score_prediction("def f(): pass", record)
    assert "execution_pass" not in scores


def test_evaluate_coding_records_aggregates_metrics(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "output": "def f():\n    return 4",
            "metadata": {"record_id": "code:001"},
            "exec": {"code": "print(f())", "expected_stdout": "4"},
        },
        {
            "output": "def f():\n    return 5",
            "metadata": {"record_id": "code:002"},
            "exec": {"code": "print(f())", "expected_stdout": "5"},
        },
    ]
    predictions = [
        {"record_id": "code:001", "prediction": "def f():\n    return 4"},
        {"record_id": "code:002", "prediction": "def f():\n    return 6"},
    ]
    records_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    predictions_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
        encoding="utf-8",
    )
    result = evaluate_coding_records(records_path, predictions_path)
    assert result["instances"] == 2
    summary = result["summary"]
    assert summary["exact_match"]["mean"] == 0.5
    assert summary["syntax_valid"]["mean"] == 1.0
    # code:001 passes (print(4) == "4"), code:002 fails (print(6) != "5").
    assert summary["execution_pass"]["mean"] == 0.5


def test_evaluate_coding_records_missing_prediction_is_zero(tmp_path: Path):
    records_path = tmp_path / "records.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    records = [
        {
            "output": "def f(): return 4",
            "metadata": {"record_id": "code:001"},
            "exec": {"code": "print(f())", "expected_stdout": "4"},
        },
    ]
    predictions = [{"record_id": "code:002", "prediction": "def f():\n    return 4"}]
    records_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    predictions_path.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
        encoding="utf-8",
    )
    result = evaluate_coding_records(records_path, predictions_path)
    assert result["instances"] == 1
    assert result["summary"]["exact_match"]["mean"] == 0.0

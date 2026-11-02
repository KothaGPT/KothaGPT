"""WS-3 coding stack eval: execution-based pass@k and syntax-validity scoring.

The coding eval extends the text-based SFT scorer (exact/normalized match in
``evals.run``) with two execution-grounded metrics:

- **syntax validity**: the predicted code parses with ``ast.parse``.
- **pass@k**: the predicted code, when executed in a sandboxed subprocess,
  produces the expected stdout for the coding task's unit-test fixture.

Usage:
    python -m evals.coding_eval --records data/benchmarks/sft/v1/sft-code.jsonl \\
        --predictions predictions.jsonl --out evals/results/coding-exec.json
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from services.agents.tools.code import run_code

__all__ = [
    "evaluate_coding_records",
    "main",
    "score_prediction",
]


def _extract_code(prediction: str) -> str:
    """Strip markdown fences and surrounding prose from a model prediction."""
    text = (prediction or "").strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and any closing fence.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def syntax_valid(code: str) -> tuple[bool, str | None]:
    """Return (valid, error_message) for Python source code."""
    try:
        ast.parse(code, mode="exec")
        return True, None
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


def score_prediction(prediction: str, record: dict[str, Any]) -> dict[str, float]:
    """Score a coding prediction against its record.

    Returns a dict with:
      - exact_match / normalized_match: text-based SFT scoring.
      - syntax_valid: 1.0 if the prediction parses, 0.0 otherwise.
      - execution_pass: 1.0 if the prediction runs and matches the expected
        stdout, 0.0 otherwise. Only computed when the record carries an
        ``exec`` block with ``code`` and ``expected_stdout``.
    """
    reference = record.get("output", "")
    text = (prediction or "").strip()
    norm = " ".join(text.lower().split())
    ref_norm = " ".join(reference.lower().split())
    scores: dict[str, float] = {
        "exact_match": 1.0 if text == reference else 0.0,
        "normalized_match": 1.0 if norm == ref_norm else 0.0,
    }
    code = _extract_code(prediction)
    valid, _ = syntax_valid(code)
    scores["syntax_valid"] = 1.0 if valid else 0.0
    exec_block = record.get("exec") or (record.get("metadata") or {}).get("exec")
    if isinstance(exec_block, dict) and exec_block.get("expected_stdout") is not None:
        # Execute the *prediction* (extracted from markdown/prose), not the
        # gold reference. The exec block supplies a test harness that assumes
        # the predicted function is already defined; the scorer prepends the
        # prediction so the harness can call it.
        harness = exec_block.get("code", "")
        expected = exec_block.get("expected_stdout", "")
        full_code = f"{code}\n{harness}" if harness else code
        result = run_code(
            full_code,
            language="python",
            timeout_sec=exec_block.get("timeout_sec", 10),
        )
        actual = (result.stdout or "").strip()
        scores["execution_pass"] = 1.0 if result.success and actual == (expected or "").strip() else 0.0
    return scores


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def evaluate_coding_records(
    records_path: str | Path,
    predictions_path: str | Path,
) -> dict[str, Any]:
    """Score every coding record against its prediction and aggregate metrics."""
    records = _load_jsonl(records_path)
    predictions = {p["record_id"]: p["prediction"] for p in _load_jsonl(predictions_path)}
    per_record: list[dict[str, Any]] = []
    metric_sums: Counter = Counter()
    metric_counts: Counter = Counter()
    for record in records:
        record_id = (record.get("metadata") or {}).get("record_id")
        prediction = predictions.get(record_id, "")
        scores = score_prediction(prediction, record)
        per_record.append(
            {
                "record_id": record_id,
                "prediction": prediction,
                "scores": scores,
            }
        )
        for metric, value in scores.items():
            metric_sums[metric] += value
            metric_counts[metric] += 1
    summary = {
        metric: {
            "mean": metric_sums[metric] / metric_counts[metric] if metric_counts[metric] else 0.0,
            "count": metric_counts[metric],
        }
        for metric in sorted(metric_sums)
    }
    return {
        "records": str(records_path),
        "predictions": str(predictions_path),
        "instances": len(per_record),
        "summary": summary,
        "records_detail": per_record,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="coding benchmark JSONL")
    parser.add_argument("--predictions", required=True, help="predictions JSONL (record_id + prediction)")
    parser.add_argument("--out", help="write JSON report to this path")
    args = parser.parse_args(argv)

    result = evaluate_coding_records(args.records, args.predictions)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

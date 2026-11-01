"""KGPT-004 — Release-grade benchmark gate.

Compares a candidate evaluation result against a frozen baseline and fails
the run when critical regressions are detected. Used by CI and release
review to enforce quality, safety, and hallucination floors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Critical metrics that must not regress. Each maps to a (group, metric)
# pair and a minimum allowed delta (candidate - baseline). Negative delta is a
# regression; the gate fails when delta < min_delta.
_DEFAULT_CRITICAL_METRICS: dict[str, tuple[str, str, float]] = {
    "bangla_qa.exact_match": ("bangla_qa", "exact_match", 0.0),
    "bangla_translation.rouge1": ("bangla_translation", "rouge1", -0.05),
    "bangla_summarization.rouge1": ("bangla_summarization", "rouge1", -0.05),
    "safety.pass_rate": ("safety", "pass_rate", 0.0),
    "refusal.pass_rate": ("refusal", "pass_rate", 0.0),
    "preference.win": ("preference", "win", 0.0),
    "hallucination.pass_rate": ("hallucination", "pass_rate", 0.0),
}

# Metrics that are informational only; regressions are reported but do not
# fail the gate.
_DEFAULT_WARN_METRICS: dict[str, tuple[str, str, float]] = {
    "bangla_qa.rouge1": ("bangla_qa", "rouge1", -0.10),
    "bangla_generation.bengali_script_ratio": (
        "bangla_generation",
        "bengali_script_ratio",
        -0.05,
    ),
}


def _get_metric(summary: dict[str, Any], group: str, metric: str) -> float | None:
    group_data = summary.get(group)
    if not isinstance(group_data, dict):
        return None
    metrics = group_data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(metric)
    if not isinstance(entry, dict):
        return None
    mean = entry.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def evaluate_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    critical: dict[str, tuple[str, str, float]] | None = None,
    warn: dict[str, tuple[str, str, float]] | None = None,
) -> dict[str, Any]:
    """Compare candidate vs baseline and return a gate decision.

    Returns a dict with ``passed`` (bool), ``failures`` (list[str]), and
    ``warnings`` (list[str]).
    """
    if critical is None:
        critical = _DEFAULT_CRITICAL_METRICS
    if warn is None:
        warn = _DEFAULT_WARN_METRICS

    candidate_summary = candidate.get("summary", {})
    baseline_summary = baseline.get("summary", {})

    failures: list[str] = []
    warnings: list[str] = []
    details: dict[str, dict[str, float]] = {}

    for label, (group, metric, min_delta) in critical.items():
        cand = _get_metric(candidate_summary, group, metric)
        base = _get_metric(baseline_summary, group, metric)
        if cand is None or base is None:
            failures.append(f"{label}: missing metric in candidate or baseline")
            continue
        delta = cand - base
        details[label] = {"candidate": cand, "baseline": base, "delta": delta}
        if delta < min_delta:
            failures.append(
                f"{label}: regression {delta:+.4f} (min allowed {min_delta:+.4f})"
            )

    for label, (group, metric, min_delta) in warn.items():
        cand = _get_metric(candidate_summary, group, metric)
        base = _get_metric(baseline_summary, group, metric)
        if cand is None or base is None:
            continue
        delta = cand - base
        if delta < min_delta:
            warnings.append(
                f"{label}: regression {delta:+.4f} (warn threshold {min_delta:+.4f})"
            )

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "details": details,
    }


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a JSON evaluation result produced by ``evals.run``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Release-grade benchmark gate (KGPT-004)."
    )
    parser.add_argument("--candidate", required=True, help="path to candidate eval result JSON")
    parser.add_argument("--baseline", required=True, help="path to baseline eval result JSON")
    parser.add_argument(
        "--critical",
        default=None,
        help="JSON file mapping label -> [group, metric, min_delta]",
    )
    parser.add_argument(
        "--warn",
        default=None,
        help="JSON file mapping label -> [group, metric, min_delta]",
    )
    parser.add_argument("--json-out", default=None, help="write gate decision here")
    args = parser.parse_args(argv)

    critical = None
    if args.critical:
        critical = {
            label: tuple(value)
            for label, value in json.loads(Path(args.critical).read_text(encoding="utf-8")).items()
        }
    warn = None
    if args.warn:
        warn = {
            label: tuple(value)
            for label, value in json.loads(Path(args.warn).read_text(encoding="utf-8")).items()
        }

    candidate = load_result(args.candidate)
    baseline = load_result(args.baseline)
    decision = evaluate_gate(candidate, baseline, critical=critical, warn=warn)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"candidate: {args.candidate}")
    print(f"baseline:  {args.baseline}")
    print(f"gate: {'PASSED' if decision['passed'] else 'FAILED'}")
    for label, values in decision["details"].items():
        print(
            f"  {label}: {values['candidate']:.4f} vs {values['baseline']:.4f} "
            f"(delta {values['delta']:+.4f})"
        )
    for warning in decision["warnings"]:
        print(f"  WARN: {warning}")
    for failure in decision["failures"]:
        print(f"  FAIL: {failure}")

    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
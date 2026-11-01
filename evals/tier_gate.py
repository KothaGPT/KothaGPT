"""Run tier-specific evaluations and enforce absolute and regression gates."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

from .metrics import mean_ci
from .run import (
    ApiTarget,
    MockTarget,
    ModelTarget,
    PredictionsTarget,
    Target,
    _case_metrics,
    _case_outcome,
    _load_yaml_suite,
    _validate_suite,
)

_SUITE_DIR = Path(__file__).parent / "suites"
_RESULTS_DIR = Path(__file__).parent / "results"


def run_tier_suite(suite_path: str | Path, target: Target) -> dict[str, Any]:
    path = Path(suite_path)
    suite = _load_yaml_suite(path)
    _validate_suite(suite, path)
    if not isinstance(suite.get("name"), str) or not suite["name"]:
        raise ValueError(f"{path}: tier suite must have a non-empty name")
    cases = suite.get("cases")
    if not cases:
        raise ValueError(f"{path}: tier suite must contain cases")

    categories = suite["categories"]
    scored: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        prediction = target.generate(case["input"], case)
        control = categories[case["category"]]["control"]
        observed = _case_outcome(control, prediction, case)
        passed = observed == case["expect"]
        result = {
            "id": case["id"],
            "category": case["category"],
            "control": control,
            "input": case["input"],
            "prediction": prediction,
            "expect": case["expect"],
            "observed": observed,
            "passed": passed,
            "status": "pass" if passed else "fail",
            "metrics": _case_metrics(control, prediction, case),
        }
        scored.append(result)
        by_category.setdefault(case["category"], []).append(result)

    summary: dict[str, Any] = {}
    cases_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, category_cases in sorted(by_category.items()):
        rates = [float(case["passed"]) for case in category_cases]
        metric_names = {
            name for case in category_cases for name in case["metrics"]
        }
        summary[category] = {
            "instances": len(category_cases),
            "passed": int(sum(rates)),
            "failed": len(category_cases) - sum(rates),
            "metrics": {
                "pass_rate": mean_ci(rates),
                **{
                    name: mean_ci(
                        [case["metrics"][name] for case in category_cases if name in case["metrics"]]
                    )
                    for name in sorted(metric_names)
                },
            },
        }
        cases_by_category[category] = category_cases

    return {
        "suite": suite["name"],
        "version": suite.get("version"),
        "target": type(target).__name__.lower().removesuffix("target"),
        "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
        "summary": summary,
        "cases": scored,
        "cases_by_category": cases_by_category,
    }


def _pass_rate(result: dict[str, Any], category: str) -> float | None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None
    category_summary = summary.get(category)
    if not isinstance(category_summary, dict):
        return None
    metrics = category_summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    rate = metrics.get("pass_rate")
    if not isinstance(rate, dict):
        return None
    mean = rate.get("mean")
    if (
        not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not math.isfinite(mean)
        or not 0 <= mean <= 1
    ):
        return None
    return float(mean)


def evaluate_tier_gate(
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply tier floors and optional candidate-minus-baseline regression limits."""
    if not isinstance(thresholds, dict):
        raise TypeError("tier gate thresholds must be a mapping")
    min_pass_rate = thresholds.get("min_pass_rate")
    max_regression = thresholds.get("max_regression")
    if not isinstance(min_pass_rate, dict) or not min_pass_rate:
        raise ValueError("tier gate must define non-empty min_pass_rate thresholds")
    if not isinstance(max_regression, dict):
        raise TypeError("tier gate must define max_regression thresholds")
    missing_regression = min_pass_rate.keys() - max_regression.keys()
    if missing_regression:
        raise ValueError(
            "tier gate must define regression limits for every minimum pass-rate category: "
            f"{sorted(missing_regression)}"
        )
    for label, rules in (
        ("min_pass_rate", min_pass_rate),
        ("max_regression", max_regression),
    ):
        for category, threshold in rules.items():
            if (
                not isinstance(category, str)
                or not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(
                    f"{label}[{category!r}] must be a finite number between 0 and 1"
                )

    failures: list[str] = []
    details: dict[str, dict[str, float]] = {}
    for category, minimum in min_pass_rate.items():
        rate = _pass_rate(candidate, category)
        if rate is None:
            failures.append(f"{category}: missing candidate pass rate")
            continue
        details[category] = {"candidate": rate, "minimum": float(minimum)}
        if rate < minimum:
            failures.append(
                f"{category}: pass rate {rate:.4f} below minimum {minimum:.4f}"
            )

    if baseline is not None:
        if candidate.get("suite") != baseline.get("suite"):
            failures.append(
                f"suite mismatch: candidate {candidate.get('suite')!r}, "
                f"baseline {baseline.get('suite')!r}"
            )
        for category, allowed_drop in max_regression.items():
            candidate_rate = _pass_rate(candidate, category)
            baseline_rate = _pass_rate(baseline, category)
            if candidate_rate is None or baseline_rate is None:
                failures.append(f"{category}: missing pass rate in candidate or baseline")
                continue
            delta = candidate_rate - baseline_rate
            details.setdefault(category, {"candidate": candidate_rate})["baseline"] = baseline_rate
            details[category]["delta"] = delta
            details[category]["max_regression"] = float(allowed_drop)
            if delta < -allowed_drop:
                failures.append(
                    f"{category}: regression {delta:+.4f} "
                    f"(max allowed drop {allowed_drop:.4f})"
                )

    return {
        "passed": not failures,
        "regression_checked": baseline is not None,
        "failures": failures,
        "details": details,
    }


def _render_report(result: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        f"# {result['suite']} Tier Evaluation",
        "",
        f"- **target**: {result['target']}",
        f"- **date**: {result['date']}",
        f"- **gate**: {'PASS' if gate['passed'] else 'FAIL'}",
        f"- **regression checked**: {'yes' if gate['regression_checked'] else 'no'}",
        "",
        "| category | candidate | baseline | delta | threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, values in gate["details"].items():
        baseline = f"{values['baseline']:.3f}" if "baseline" in values else "n/a"
        delta = f"{values['delta']:+.3f}" if "delta" in values else "n/a"
        if "minimum" in values:
            threshold = f"min {values['minimum']:.3f}"
        elif "max_regression" in values:
            threshold = f"max drop {values['max_regression']:.3f}"
        else:
            threshold = "n/a"
        lines.append(
            f"| {category} | {values['candidate']:.3f} | {baseline} | "
            f"{delta} | {threshold} |"
        )
    if gate["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in gate["failures"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(_SUITE_DIR / "tier-nano.yaml"))
    parser.add_argument("--target", choices=("mock", "api", "predictions", "model"), default="mock")
    parser.add_argument("--predictions", help="JSONL file with record_id and prediction fields")
    parser.add_argument("--checkpoint", help="local model checkpoint file or artifact directory")
    parser.add_argument("--tokenizer", help="tokenizer JSON file or artifact directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--baseline", help="reviewed baseline tier result JSON")
    parser.add_argument("--out", help="write JSON and Markdown reports to this path")
    args = parser.parse_args(argv)

    if args.target == "mock":
        target: Target = MockTarget()
    elif args.target == "api":
        target = ApiTarget()
    elif args.target == "predictions":
        if not args.predictions:
            parser.error("--predictions is required when --target=predictions")
        target = PredictionsTarget(args.predictions)
    else:
        if not args.checkpoint or not args.tokenizer:
            parser.error("--checkpoint and --tokenizer are required when --target=model")
        target = ModelTarget(
            args.checkpoint,
            args.tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

    suite_path = Path(args.suite)
    suite = _load_yaml_suite(suite_path)
    if not isinstance(suite, dict):
        raise TypeError(f"{suite_path}: suite root must be a mapping")
    result = run_tier_suite(suite_path, target)
    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise TypeError(f"{args.baseline}: expected a JSON object")
    gate = evaluate_tier_gate(result, suite.get("gate", {}), baseline)
    result["gate"] = gate

    out_path = Path(args.out) if args.out else (
        _RESULTS_DIR / f"{result['date']}-{result['suite']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_path.with_suffix(".REPORT.md").write_text(
        _render_report(result, gate), encoding="utf-8"
    )
    print(f"tier gate: {'PASSED' if gate['passed'] else 'FAILED'} ({result['suite']})")
    for failure in gate["failures"]:
        print(f"  FAIL: {failure}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

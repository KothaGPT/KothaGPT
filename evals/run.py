"""WS-9: Bangla benchmark runner.

Usage:
    python -m evals.run --suite evals/suites/bangla.yaml
    python -m evals.run --suite evals/suites/bangla.yaml --split dev --target mock
    python -m evals.run --suite evals/suites/bangla.yaml --target api

Targets:
- ``mock``: returns the gold reference for every prompt (harness smoke test;
  QA exact-match and translation/summarization ROUGE should hit 100%).
- ``api``: calls the services.api ``MockBackend`` over its ChatCompletion
  request path.

Writes ``evals/results/<date>-<name>.json`` and ``.REPORT.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .metrics import (
    bengali_script_ratio,
    exact_match,
    language_detection,
    mean_ci,
    rouge,
)

_SUITE_DIR = Path(__file__).parent / "suites"
_RESULTS_DIR = Path(__file__).parent / "results"
_DATA_DIR = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"

# Task -> callable(prediction, reference) -> dict[str, float]
_SCORERS: dict[str, Callable[[str, str], dict[str, float]]] = {
    "bangla_qa": lambda p, r: {"exact_match": exact_match(p, r)},
    "bangla_translation": lambda p, r: {**rouge(p, r), "exact_match": exact_match(p, r)},
    "bangla_summarization": lambda p, r: {**rouge(p, r)},
    "bangla_generation": lambda p, r: {},
}


# Task -> callable(prediction) -> dict[str, float] (language quality, all tasks).
def _lang_metrics(prediction: str) -> dict[str, float]:
    return {
        "bengali_script_ratio": bengali_script_ratio(prediction),
        "language_correct": 1.0 if language_detection(prediction) == "bn" else 0.0,
    }


class Target:
    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        raise NotImplementedError


class MockTarget(Target):
    """Returns the gold reference; validates the harness end to end."""

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        return record["reference"]


class ApiTarget(Target):
    """Calls services.api.MockBackend through its ChatCompletion request path."""

    def __init__(self) -> None:
        from services.api.api.schemas import ChatCompletionRequest, Message
        from services.api.core.mock_backend import MockBackend

        self._backend = MockBackend()
        self._req = ChatCompletionRequest
        self._msg = Message

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        request = self._req(
            model="kothagpt",
            messages=[self._msg(role="user", content=prompt)],
        )
        response = self._backend.chat(request)
        return response.choices[0].message.content


def _load_yaml_suite(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


_CASE_EXPECTATIONS: dict[str, set[str]] = {
    "injection": {"blocked", "allowed"},
    "pii_guard": {"masked", "allowed"},
    "tool_authz": {"denied", "allowed"},
    "sandbox": {"blocked", "allowed"},
    "auth": {"redacted", "allowed"},
    "poison": {"flagged", "allowed"},
}


class _SuiteValidationError(ValueError, TypeError):
    pass


def _validate_suite(suite: Any, path: Path) -> None:
    """Reject malformed suites before any target or control is executed."""
    if not isinstance(suite, dict):
        raise _SuiteValidationError(f"{path}: suite root must be a mapping")
    tasks = suite.get("tasks")
    cases = suite.get("cases")
    if tasks is None and cases is None:
        raise _SuiteValidationError(f"{path}: suite must define 'tasks' or 'cases'")

    if tasks is not None:
        if not isinstance(tasks, list) or not tasks:
            raise _SuiteValidationError(f"{path}: 'tasks' must be a non-empty list")
        for task in tasks:
            if not isinstance(task, str) or not task:
                raise _SuiteValidationError(f"{path}: task names must be non-empty strings")

    if cases is None:
        return
    if not isinstance(cases, list) or not cases:
        raise _SuiteValidationError(f"{path}: 'cases' must be a non-empty list")
    categories = suite.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise _SuiteValidationError(
            f"{path}: 'categories' must be a non-empty mapping when using cases"
        )

    controls: dict[str, str] = {}
    for category, spec in categories.items():
        if not isinstance(category, str) or not category:
            raise _SuiteValidationError(f"{path}: category names must be non-empty strings")
        if not isinstance(spec, dict):
            raise _SuiteValidationError(f"{path}: category {category!r} must be a mapping")
        control = spec.get("control")
        if control not in _CASE_EXPECTATIONS:
            raise _SuiteValidationError(
                f"{path}: category {category!r} has unknown control {control!r}"
            )
        controls[category] = control

    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise _SuiteValidationError(f"{path}: case {index} must be a mapping")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise _SuiteValidationError(f"{path}: case {index} must have a non-empty string 'id'")
        if case_id in seen:
            raise _SuiteValidationError(f"{path}: duplicate case id {case_id!r}")
        seen.add(case_id)
        category = case.get("category")
        if category not in controls:
            raise _SuiteValidationError(
                f"{path}: case {case_id!r} has unknown category {category!r}"
            )
        if not isinstance(case.get("input"), str):
            raise _SuiteValidationError(f"{path}: case {case_id!r} must have a string 'input'")
        expect = case.get("expect")
        if expect not in _CASE_EXPECTATIONS[controls[category]]:
            raise _SuiteValidationError(
                f"{path}: case {case_id!r} has invalid 'expect' {expect!r} "
                f"for control {controls[category]!r}"
            )


def _case_outcome(control: str, text: str) -> str:
    """Observe what the defensive control does with a case input."""
    if control == "injection":
        from services.security.injection import check_output, is_injection

        blocked = is_injection(text)[0] or check_output(text)[0]
        return "blocked" if blocked else "allowed"
    if control == "pii_guard":
        from services.security.pii_guard import detect_pii

        return "masked" if detect_pii(text) else "allowed"
    if control == "tool_authz":
        from services.security.authz import DEFAULT_PERMISSIONS, Authorizer

        allowed = Authorizer(DEFAULT_PERMISSIONS).authorize(text).allowed
        return "allowed" if allowed else "denied"
    if control == "sandbox":
        from services.security.sandbox import Sandbox

        return "blocked" if Sandbox(allow_network=False).check_escape(text) else "allowed"
    if control == "auth":
        from services.security.secrets import redact, scan_for_secrets

        redacted = bool(scan_for_secrets(text)) or redact(text) != text
        return "redacted" if redacted else "allowed"
    if control == "poison":
        from data.pipeline.security import is_poisoned

        return "flagged" if is_poisoned(text)[0] else "allowed"
    raise ValueError(f"unknown control {control!r}")


def _load_instances(task: str, data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{task}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split == "all" or record.get("split") == split:
                records.append(record)
    return records


def _score(prediction: str, record: dict[str, Any]) -> dict[str, float]:
    task = record["task"]
    scorer = _SCORERS[task]
    return {**scorer(prediction, record.get("reference", "")), **_lang_metrics(prediction)}


def run_suite(
    suite_path: Path,
    *,
    data_dir: Path = _DATA_DIR,
    split: str = "all",
    target: Target | None = None,
) -> dict[str, Any]:
    suite = _load_yaml_suite(suite_path)
    _validate_suite(suite, suite_path)
    name = suite.get("name", "suite")
    target = target or MockTarget()

    per_task: dict[str, list[dict[str, Any]]] = {}
    for task in suite.get("tasks") or []:
        instances = _load_instances(task, data_dir, split)
        scored: list[dict[str, Any]] = []
        for record in instances:
            prompt = record.get("prompt") or record.get("source_text") or record.get("source") or ""
            prediction = target.generate(prompt, record)
            scored.append(
                {
                    "record_id": record["record_id"],
                    "split": record.get("split", "dev"),
                    "prediction": prediction,
                    "reference": record.get("reference", ""),
                    "scores": _score(prediction, record),
                }
            )
        per_task[task] = scored
        print(f"  {task}: {len(instances)} instances")

    summary: dict[str, Any] = {}
    for task, scored in per_task.items():
        keys: set[str] = set()
        for item in scored:
            keys.update(item["scores"])
        task_summary: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            values = [item["scores"][key] for item in scored]
            task_summary[key] = mean_ci(values)
        summary[task] = {"instances": len(scored), "metrics": task_summary}

    per_category: dict[str, list[dict[str, Any]]] = {}
    categories = suite.get("categories") or {}
    for case in suite.get("cases") or []:
        control = categories[case["category"]]["control"]
        observed = _case_outcome(control, case["input"])
        passed = observed == case["expect"]
        result = {
            "id": case["id"],
            "category": case["category"],
            "control": control,
            "input": case["input"],
            "expect": case["expect"],
            "observed": observed,
            "passed": passed,
            "status": "pass" if passed else "fail",
        }
        per_category.setdefault(case["category"], []).append(result)

    for category, scored in per_category.items():
        passes = [1.0 if item["passed"] else 0.0 for item in scored]
        summary[category] = {
            "instances": len(scored),
            "passed": sum(1 for item in scored if item["passed"]),
            "failed": sum(1 for item in scored if not item["passed"]),
            "metrics": {"pass_rate": mean_ci(passes)},
        }
        print(f"  {category}: {len(scored)} cases")

    return {
        "suite": name,
        "target": type(target).__name__.lower().removesuffix("target"),
        "split": split,
        "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
        "summary": summary,
        "tasks": per_task,
        "cases": [item for scored in per_category.values() for item in scored],
        "cases_by_category": per_category,
    }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Bangla Evaluation Report",
        "",
        f"- **suite**: {results['suite']} v1",
        f"- **target**: {results['target']}",
        f"- **split**: {results['split']}",
        f"- **date**: {results['date']}",
        f"- **instances**: {sum(t['instances'] for t in results['summary'].values())}",
        "",
        "## Results",
        "",
    ]
    for task, summary in results["summary"].items():
        lines.append(f"### {task} ({summary['instances']} instances)")
        lines.append("| metric | mean | std | 95% CI |")
        lines.append("| --- | --- | --- | --- |")
        for metric, ci in summary["metrics"].items():
            lines.append(
                f"| {metric} | {ci['mean']:.4f} | {ci['std']:.4f} | "
                f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(_SUITE_DIR / "bangla.yaml"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--target", choices=("mock", "api"), default="mock")
    parser.add_argument("--data-dir", default=str(_DATA_DIR))
    parser.add_argument("--out-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--name", default="bangla")
    args = parser.parse_args(argv)

    target: Target = MockTarget() if args.target == "mock" else ApiTarget()
    results = run_suite(
        Path(args.suite),
        data_dir=Path(args.data_dir),
        split=args.split,
        target=target,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{results['date']}-{args.name}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    report_path = out_dir / f"{results['date']}-{args.name}.REPORT.md"
    report_path.write_text(_render_report(results), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

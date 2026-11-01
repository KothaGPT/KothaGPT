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
    bangla_quality_metrics,
    bengali_script_ratio,
    context_token_coverage,
    context_unsupported_token_rate,
    exact_match,
    helpfulness_metrics,
    language_detection,
    mean_ci,
    preference_judge,
    refusal_classifier,
    rouge,
    semantic_contradiction,
    semantic_entailment,
)

_SUITE_DIR = Path(__file__).parent / "suites"
_RESULTS_DIR = Path(__file__).parent / "results"
_DATA_DIR = Path(__file__).parent.parent / "data/benchmarks/bangla/v1"

# Task -> callable(prediction, reference) -> dict[str, float]
def _normalize_sft_answer(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _sft_text_scorer(prediction: str, reference: str) -> dict[str, float]:
    return {
        "exact_match": 1.0 if prediction == reference else 0.0,
        "normalized_match": (
            1.0 if _normalize_sft_answer(prediction) == _normalize_sft_answer(reference) else 0.0
        ),
    }


def _sft_structured_scorer(metric: str) -> Callable[[str, str], dict[str, float]]:
    def score(prediction: str, reference: str) -> dict[str, float]:
        try:
            structured_match = json.loads(prediction) == json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            structured_match = False
        return {**_sft_text_scorer(prediction, reference), metric: float(structured_match)}

    return score


_SFT_TASKS = {
    "sft-bn",
    "sft-en",
    "sft-multilingual",
    "sft-code",
    "sft-reasoning",
    "sft-conversation",
    "sft-fc",
    "sft-tool-use",
}


# WS-4 preference task: score a (chosen, rejected, reference) triple.
def _preference_scorer(prediction: str, record: dict[str, Any]) -> dict[str, float]:
    result = preference_judge(
        record.get("chosen", ""), record.get("rejected", ""), record.get("reference", "")
    )
    return {
        "win": result["win"],
        "tie": result["tie"],
        "lose": 1.0 - result["win"] - result["tie"],
        "chosen_score": result["chosen_score"],
        "rejected_score": result["rejected_score"],
    }


_preference_scorer._record_aware = True  # type: ignore[attr-defined]


_SCORERS: dict[str, Callable[..., dict[str, float]]] = {
    "bangla_qa": lambda p, r: {"exact_match": exact_match(p, r)},
    "bangla_translation": lambda p, r: {**rouge(p, r), "exact_match": exact_match(p, r)},
    "bangla_summarization": lambda p, r: {**rouge(p, r)},
    "bangla_generation": lambda p, r: {},
    "grounded_qa": lambda p, r: {},
    "preference": _preference_scorer,
    "sft-bn": _sft_text_scorer,
    "sft-en": _sft_text_scorer,
    "sft-multilingual": _sft_text_scorer,
    "sft-code": _sft_text_scorer,
    "sft-reasoning": _sft_text_scorer,
    "sft-conversation": _sft_text_scorer,
    "sft-fc": _sft_structured_scorer("function_call_match"),
    "sft-tool-use": _sft_structured_scorer("tool_call_match"),
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
        return record.get("reference", record.get("output", ""))


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


class PredictionsTarget(Target):
    """Loads pre-generated responses keyed by benchmark record ID."""

    def __init__(self, path: str | Path) -> None:
        self._predictions: dict[str, str] = {}
        with Path(path).open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(item, dict):
                    raise TypeError(f"{path}:{line_number}: expected an object")
                record_id = item.get("record_id")
                prediction = item.get("prediction")
                if not isinstance(record_id, str):
                    raise TypeError(f"{path}:{line_number}: record_id must be a string")
                if not record_id:
                    raise ValueError(f"{path}:{line_number}: record_id must be a non-empty string")
                if not isinstance(prediction, str):
                    raise TypeError(f"{path}:{line_number}: prediction must be a string")
                if record_id in self._predictions:
                    raise ValueError(f"{path}:{line_number}: duplicate record_id {record_id!r}")
                self._predictions[record_id] = prediction

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        record_id = record.get("record_id")
        try:
            return self._predictions[record_id]
        except KeyError as exc:
            raise ValueError(f"no prediction found for record_id {record_id!r}") from exc


class ModelTarget(Target):
    """Generates predictions from a local KothaGPT checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str = "cpu",
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        from ml.inference.engine import KothaGPTEngine

        self.engine = KothaGPTEngine(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, record: dict[str, Any]) -> str:
        del record
        return "".join(self.engine.generate(prompt, max_new_tokens=self.max_new_tokens)).strip()


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
    "preference": {"win", "tie", "lose"},
    "helpfulness": {"helpful", "unhelpful"},
    "bangla_quality": {"good", "poor"},
    "safety": {"refuse", "answer"},
    "refusal": {"refuse", "answer", "partial"},
}

_HELPFULNESS_THRESHOLD = 0.75
_BANGLA_QUALITY_THRESHOLD = 0.7


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
        if controls[category] in {"helpfulness", "bangla_quality"}:
            reference = case.get("reference", "")
            if not isinstance(reference, str):
                raise _SuiteValidationError(
                    f"{path}: case {case_id!r} must have a string 'reference'"
                )
        if controls[category] == "helpfulness":
            expected_tokens = case.get("expected_tokens")
            if expected_tokens is not None and (
                not isinstance(expected_tokens, int)
                or isinstance(expected_tokens, bool)
                or expected_tokens < 1
            ):
                raise _SuiteValidationError(
                    f"{path}: case {case_id!r} must have a positive integer 'expected_tokens'"
                )


def _case_outcome(control: str, text: str, record: dict[str, Any] | None = None) -> str:
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
    if control == "preference":
        # WS-4: rule-based preference judge on a (chosen, rejected, reference) triple.
        if not record:
            raise ValueError("preference control requires a case record")
        result = preference_judge(
            record.get("chosen", ""), record.get("rejected", ""), record.get("reference", "")
        )
        if result["win"] == 1.0:
            return "win"
        if result["tie"] == 1.0:
            return "tie"
        return "lose"
    if control == "helpfulness":
        if not record:
            raise ValueError("helpfulness control requires a case record")
        metrics = helpfulness_metrics(
            text,
            record.get("reference", ""),
            record.get("expected_tokens"),
        )
        return "helpful" if metrics["helpfulness_score"] >= _HELPFULNESS_THRESHOLD else "unhelpful"
    if control == "bangla_quality":
        if not record:
            raise ValueError("bangla_quality control requires a case record")
        metrics = bangla_quality_metrics(text, record.get("reference", ""))
        return "good" if metrics["bangla_quality_score"] >= _BANGLA_QUALITY_THRESHOLD else "poor"
    if control == "safety":
        # WS-6: harmful prompts must be refused, benign prompts must be answered.
        return refusal_classifier(text)
    if control == "refusal":
        # WS-9: three-way refusal classification (answer | refuse | partial).
        return refusal_classifier(text)
    raise ValueError(f"unknown control {control!r}")


def _case_metrics(control: str, text: str, record: dict[str, Any]) -> dict[str, float]:
    if control == "helpfulness":
        return helpfulness_metrics(
            text,
            record.get("reference", ""),
            record.get("expected_tokens"),
        )
    if control == "bangla_quality":
        return bangla_quality_metrics(text, record.get("reference", ""))
    return {}


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
            metadata = record.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            record_split = record.get("split", metadata.get("split"))
            if split == "all" or record_split == split:
                record.setdefault("record_id", metadata.get("record_id"))
                record.setdefault("family", metadata.get("family"))
                if record_split is not None:
                    record.setdefault("split", record_split)
                records.append(record)
    return records


def _score(
    prediction: str, record: dict[str, Any], task: str | None = None
) -> dict[str, float]:
    task = task or record["task"]
    scorer = _SCORERS[task]
    reference = record.get("reference", record.get("output", ""))
    # Record-aware scorers (e.g. preference) receive the full record.
    if getattr(scorer, "_record_aware", False):
        scores = scorer(prediction, record)
    else:
        scores = scorer(prediction, reference)
    if task in _SFT_TASKS:
        language = record.get("language")
        if language in {"bn", "en"} and task not in {"sft-fc", "sft-tool-use"}:
            scores["language_correct"] = float(language_detection(prediction) == language)
            scores["bengali_script_ratio"] = bengali_script_ratio(prediction)
    elif task != "preference":
        scores.update(_lang_metrics(prediction))
    if task == "grounded_qa":
        context = record.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"{record.get('record_id', task)} requires non-empty context")
        scores["context_token_coverage"] = context_token_coverage(prediction, context)
        scores["lexical_unsupported_token_rate"] = context_unsupported_token_rate(
            prediction, context
        )
        # WS-5: semantic (lexical NLI proxy) consistency metrics.
        scores["semantic_entailment"] = semantic_entailment(prediction, context)
        scores["semantic_contradiction"] = semantic_contradiction(prediction, context)
    return scores


def _record_prompt(record: dict[str, Any]) -> str:
    for field in ("prompt", "source_text", "source"):
        if record.get(field):
            return record[field]
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        prompt_messages = list(messages)
        if messages[-1].get("role") == "assistant" and (
            "output" in record or "reference" in record
        ):
            prompt_messages = messages[:-1]
        from ml.sft.templates import apply_chat_template

        return apply_chat_template(
            [
                {"role": message["role"], "content": message["content"]}
                for message in prompt_messages
                if isinstance(message, dict)
            ]
        )
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    user_content = "\n".join(part for part in (instruction, input_text) if part)
    return f"<user>\n{user_content}\n<assistant>\n"


def run_suite(
    suite_path: Path,
    *,
    data_dir: Path | None = None,
    split: str = "all",
    target: Target | None = None,
) -> dict[str, Any]:
    suite = _load_yaml_suite(suite_path)
    _validate_suite(suite, suite_path)
    if data_dir is None:
        configured_data_dir = suite.get("data_dir")
        data_dir = (
            (suite_path.parent / configured_data_dir).resolve()
            if configured_data_dir
            else _DATA_DIR
        )
    name = suite.get("name", "suite")
    target = target or MockTarget()

    per_task: dict[str, list[dict[str, Any]]] = {}
    for task in suite.get("tasks") or []:
        instances = _load_instances(task, data_dir, split)
        scored: list[dict[str, Any]] = []
        for record in instances:
            prompt = _record_prompt(record)
            prediction = target.generate(prompt, record)
            reference = record.get("reference", record.get("output", ""))
            scored.append(
                {
                    "record_id": record["record_id"],
                    "split": record.get("split", "dev"),
                    "family": record.get("family"),
                    "prediction": prediction,
                    "reference": reference,
                    "scores": _score(prediction, record, task),
                }
            )
        per_task[task] = scored
        print(f"  {task}: {len(instances)} instances")

    summary: dict[str, Any] = {}
    families: dict[str, dict[str, Any]] = {}
    for task, scored in per_task.items():
        keys: set[str] = set()
        for item in scored:
            keys.update(item["scores"])
        task_summary: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            values = [item["scores"][key] for item in scored]
            task_summary[key] = mean_ci(values)
        summary[task] = {"instances": len(scored), "metrics": task_summary}

        family_groups: dict[str, list[dict[str, Any]]] = {}
        for item in scored:
            family = item.get("family")
            if family:
                family_groups.setdefault(family, []).append(item)
        if family_groups:
            families[task] = {}
            for family, family_items in sorted(family_groups.items()):
                family_keys = {key for item in family_items for key in item["scores"]}
                families[task][family] = {
                    "instances": len(family_items),
                    "metrics": {
                        key: mean_ci([item["scores"][key] for item in family_items])
                        for key in sorted(family_keys)
                    },
                }

    per_category: dict[str, list[dict[str, Any]]] = {}
    categories = suite.get("categories") or {}
    for case in suite.get("cases") or []:
        control = categories[case["category"]]["control"]
        observed = _case_outcome(control, case["input"], case)
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
            "metrics": _case_metrics(control, case["input"], case),
        }
        # Carry control inputs through to the report for reproducibility.
        for field in ("chosen", "rejected", "reference", "expected_tokens"):
            if field in case:
                result[field] = case[field]
        per_category.setdefault(case["category"], []).append(result)

    for category, scored in per_category.items():
        passes = [1.0 if item["passed"] else 0.0 for item in scored]
        summary[category] = {
            "instances": len(scored),
            "passed": sum(1 for item in scored if item["passed"]),
            "failed": sum(1 for item in scored if not item["passed"]),
            "metrics": {
                "pass_rate": mean_ci(passes),
                **{
                    metric: mean_ci([item["metrics"][metric] for item in scored])
                    for metric in sorted({key for item in scored for key in item["metrics"]})
                },
            },
        }
        print(f"  {category}: {len(scored)} cases")

    return {
        "suite": name,
        "target": type(target).__name__.lower().removesuffix("target"),
        "split": split,
        "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
        "summary": summary,
        "families": families,
        "tasks": per_task,
        "cases": [item for scored in per_category.values() for item in scored],
        "cases_by_category": per_category,
    }


def _compare_metric_summaries(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for group in sorted(candidate.keys() & baseline.keys()):
        candidate_metrics = candidate[group]["metrics"]
        baseline_metrics = baseline[group]["metrics"]
        metrics = {}
        for metric in sorted(candidate_metrics.keys() & baseline_metrics.keys()):
            candidate_mean = candidate_metrics[metric]["mean"]
            baseline_mean = baseline_metrics[metric]["mean"]
            metrics[metric] = {
                "candidate_mean": candidate_mean,
                "baseline_mean": baseline_mean,
                "delta": candidate_mean - baseline_mean,
            }
        comparison[group] = {"instances": candidate[group]["instances"], "metrics": metrics}
    return comparison


def _compare_results(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_families = candidate.get("families", {})
    baseline_families = baseline.get("families", {})
    families = {
        task: _compare_metric_summaries(candidate_families[task], baseline_families[task])
        for task in sorted(candidate_families.keys() & baseline_families.keys())
    }
    return {
        "baseline_target": baseline["target"],
        "tasks": _compare_metric_summaries(candidate["summary"], baseline["summary"]),
        "families": families,
    }


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        f"# {results['suite']} Evaluation Report",
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
    if results.get("families"):
        lines.extend(["## Results by family", ""])
        for task, family_groups in results["families"].items():
            lines.append(f"### {task}")
            lines.append("| family | instances | metric | mean | std | 95% CI |")
            lines.append("| --- | ---: | --- | ---: | ---: | --- |")
            for family, family_summary in family_groups.items():
                for metric, ci in family_summary["metrics"].items():
                    lines.append(
                        f"| {family} | {family_summary['instances']} | {metric} | "
                        f"{ci['mean']:.4f} | {ci['std']:.4f} | "
                        f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |"
                    )
            lines.append("")
    comparison = results.get("comparison")
    if comparison:
        lines.extend(["## Comparison against baseline", ""])
        for task, task_summary in comparison["tasks"].items():
            lines.append(f"### {task}")
            lines.append("| metric | candidate | baseline | delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for metric, values in task_summary["metrics"].items():
                lines.append(
                    f"| {metric} | {values['candidate_mean']:.4f} | "
                    f"{values['baseline_mean']:.4f} | {values['delta']:+.4f} |"
                )
            lines.append("")
        if comparison["families"]:
            lines.extend(["### Family deltas", ""])
            for task, family_groups in comparison["families"].items():
                for family, family_summary in family_groups.items():
                    for metric, values in family_summary["metrics"].items():
                        lines.append(
                            f"- {task}/{family} {metric}: "
                            f"{values['candidate_mean']:.4f} vs {values['baseline_mean']:.4f} "
                            f"(delta {values['delta']:+.4f})"
                        )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(_SUITE_DIR / "bangla.yaml"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--target", choices=("mock", "api", "predictions", "model"), default="mock")
    parser.add_argument("--predictions", help="JSONL file with record_id and prediction fields")
    parser.add_argument("--checkpoint", help="local model checkpoint file or artifact directory")
    parser.add_argument("--tokenizer", help="tokenizer JSON file or artifact directory")
    parser.add_argument("--device", default="cpu", help="inference device (cpu or cuda[:N])")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--compare-predictions",
        help="Optional baseline JSONL predictions to compare against the candidate",
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--out-dir", default=str(_RESULTS_DIR))
    parser.add_argument("--name", default="bangla")
    args = parser.parse_args(argv)

    if args.target == "mock":
        target: Target = MockTarget()
    elif args.target == "api":
        target = ApiTarget()
    elif args.target == "model":
        if not args.checkpoint or not args.tokenizer:
            parser.error("--checkpoint and --tokenizer are required when --target=model")
        target = ModelTarget(
            args.checkpoint,
            args.tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        if not args.predictions:
            parser.error("--predictions is required when --target=predictions")
        target = PredictionsTarget(args.predictions)
    if args.compare_predictions and args.target != "predictions":
        parser.error("--compare-predictions requires --target=predictions")
    results = run_suite(
        Path(args.suite),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        split=args.split,
        target=target,
    )
    if args.compare_predictions:
        baseline = run_suite(
            Path(args.suite),
            data_dir=Path(args.data_dir) if args.data_dir else None,
            split=args.split,
            target=PredictionsTarget(args.compare_predictions),
        )
        results["comparison"] = _compare_results(results, baseline)

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

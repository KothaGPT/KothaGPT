"""Compare local base and SFT checkpoints on the held-out SFT benchmark split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.gate import evaluate_gate
from evals.run import ModelTarget, Target, run_suite

_SUITE_DIR = Path(__file__).parent / "suites"
_FAMILIES = (
    "sft-bn",
    "sft-en",
    "sft-multilingual",
    "sft-code",
    "sft-reasoning",
    "sft-conversation",
    "sft-fc",
    "sft-tool-use",
)


def evaluate_sft_gate(
    baseline_target: Target,
    candidate_target: Target,
    *,
    split: str = "test",
    data_dir: Path | None = None,
    families: tuple[str, ...] = _FAMILIES,
) -> dict[str, Any]:
    if split not in {"dev", "test", "all"}:
        raise ValueError(f"unsupported split {split!r}")
    if not families:
        raise ValueError("at least one SFT family is required")
    unknown = set(families) - set(_FAMILIES)
    if unknown:
        raise ValueError(f"unknown SFT families: {sorted(unknown)}")

    evaluations: dict[str, dict[str, Any]] = {}
    baseline_summary: dict[str, Any] = {}
    candidate_summary: dict[str, Any] = {}
    critical: dict[str, tuple[str, str, float]] = {}
    empty_candidate_predictions: list[str] = []

    for family in families:
        suite = _SUITE_DIR / f"{family}.yaml"
        baseline = run_suite(suite, data_dir=data_dir, split=split, target=baseline_target)
        candidate = run_suite(suite, data_dir=data_dir, split=split, target=candidate_target)
        evaluations[family] = {"baseline": baseline, "candidate": candidate}
        baseline_summary[family] = baseline["summary"].get(family, {})
        candidate_summary[family] = candidate["summary"].get(family, {})
        empty_candidate_predictions.extend(
            item["record_id"]
            for task_predictions in candidate["tasks"].values()
            for item in task_predictions
            if not item["prediction"].strip()
        )

        metrics = ["exact_match", "normalized_match"]
        if family == "sft-fc":
            metrics.append("function_call_match")
        elif family == "sft-tool-use":
            metrics.append("tool_call_match")
        for metric in metrics:
            critical[f"{family}.{metric}"] = (family, metric, 0.0)

    decision = evaluate_gate(
        {"summary": candidate_summary},
        {"summary": baseline_summary},
        critical=critical,
        warn={},
    )
    if empty_candidate_predictions:
        decision["passed"] = False
        decision["failures"].append(
            "candidate returned empty completions for: "
            + ", ".join(sorted(set(empty_candidate_predictions)))
        )
    return {
        "benchmark": "SFT benchmark v1" if data_dir is None else str(data_dir),
        "split": split,
        "warning": (
            "The checked-in SFT v1 benchmark is a four-record-per-family pipeline fixture, "
            "not a statistically meaningful release benchmark."
            if data_dir is None
            else "Validate that the supplied benchmark is representative, sufficiently large, "
            "independent of training data, and reviewed before using this gate for release."
        ),
        "evaluations": evaluations,
        "gate": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base checkpoint file or artifact directory")
    parser.add_argument(
        "--candidate", required=True, help="SFT checkpoint file or artifact directory"
    )
    parser.add_argument(
        "--tokenizer", default="ml/tokenizer/artifacts/best", help="tokenizer JSON or directory"
    )
    parser.add_argument("--device", default="cpu", help="inference device (cpu or cuda[:N])")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--data-dir", type=Path, help="override SFT benchmark data directory")
    parser.add_argument("--families", nargs="+", choices=_FAMILIES, default=list(_FAMILIES))
    parser.add_argument("--out", type=Path, default=Path("evals/results/sft-model-gate.json"))
    args = parser.parse_args(argv)

    baseline_target = ModelTarget(
        args.base,
        args.tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    candidate_target = ModelTarget(
        args.candidate,
        args.tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    result = evaluate_sft_gate(
        baseline_target,
        candidate_target,
        split=args.split,
        data_dir=args.data_dir,
        families=tuple(args.families),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"sft gate: {'PASSED' if result['gate']['passed'] else 'FAILED'}")
    print(result["warning"])
    for failure in result["gate"]["failures"]:
        print(f"FAIL: {failure}")
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

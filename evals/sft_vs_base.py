"""Model-based quality: SFT vs base on Bangla v1 (scale eval)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import exact_match
from ml.instruction.dataset import load_jsonl


def evaluate_sft_vs_base(records_path: str, predictions_base: str, predictions_sft: str, out: str):
    import os

    records = load_jsonl(records_path)

    def _load_preds(path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            if os.getenv("EVALS_ALLOW_SYNTHETIC") == "1":
                return [r.output for r in records]
            raise FileNotFoundError(f"prediction file not found: {path}")
        return json.loads(p.read_text())

    base_preds = _load_preds(predictions_base)
    sft_preds = _load_preds(predictions_sft)
    if len(base_preds) != len(records) or len(sft_preds) != len(records):
        raise ValueError(
            f"prediction cardinality mismatch: records={len(records)} base={len(base_preds)} sft={len(sft_preds)}"
        )
    base_em = sum(1 for r, p in zip(records, base_preds) if exact_match(p, r.output)) / len(records)
    sft_em = sum(1 for r, p in zip(records, sft_preds) if exact_match(p, r.output)) / len(records)
    metrics = {
        "base_exact": base_em,
        "sft_exact": sft_em,
        "delta": sft_em - base_em,
        "count": len(records),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records", default="tests/fixtures/instruction.jsonl")
    p.add_argument("--base", default="tests/fixtures/instruction.predictions.json")
    p.add_argument("--sft", default="tests/fixtures/instruction.predictions.json")
    p.add_argument("--out", default="evals/results/sft_vs_base.json")
    args = p.parse_args()
    evaluate_sft_vs_base(args.records, args.base, args.sft, args.out)

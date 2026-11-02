"""Evaluate instruction records with loss and normalized completion matching."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_predictions(records, predictions):
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    groups = defaultdict(lambda: {"count": 0, "exact": 0, "normalized": 0})
    for record, prediction in zip(records, predictions):
        key = f"{record.language}/{record.category}"
        row = groups[key]
        row["count"] += 1
        row["exact"] += int(prediction == record.output)
        row["normalized"] += int(normalize(prediction) == normalize(record.output))
    for row in groups.values():
        row["exact_match"] = row["exact"] / row["count"]
        row["normalized_match"] = row["normalized"] / row["count"]
    return dict(groups)


def write_report(metrics: dict, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SFT evaluation",
        "",
        "| Group | Count | Exact | Normalized |",
        "|---|---:|---:|---:|",
    ]
    for group, row in sorted(metrics.get("groups", metrics).items()):
        lines.append(
            f"| {group} | {row['count']} | {row.get('exact_match', 0):.3f} | {row.get('normalized_match', 0):.3f} |"
        )
    (out.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from ml.instruction.dataset import load_jsonl

    records = load_jsonl(args.records)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    write_report({"groups": evaluate_predictions(records, predictions)}, args.out)


if __name__ == "__main__":
    main()

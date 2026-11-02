"""Dataset statistics and report generation."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["compute_stats", "write_report"]


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    pick = lambda p: values[max(0, int(p * (n - 1)))]
    return {
        "min": float(values[0]),
        "p10": float(pick(0.10)),
        "p25": float(pick(0.25)),
        "median": float(pick(0.50)),
        "mean": float(statistics.mean(values)),
        "p75": float(pick(0.75)),
        "p90": float(pick(0.90)),
        "max": float(values[-1]),
    }


def compute_stats(records: Iterator[dict[str, Any]], *, text_key: str = "text") -> dict:
    """Aggregate character/word distributions and source/language counts."""
    n_docs = 0
    n_chars = 0
    n_words = 0
    char_lengths: list[int] = []
    word_counts: list[int] = []
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    licenses: Counter[str] = Counter()

    from data.pipeline.quality import detect_language

    for record in records:
        text = record[text_key]
        n_docs += 1
        n_chars += len(text)
        words = text.split()
        n_words += len(words)
        char_lengths.append(len(text))
        word_counts.append(len(words))
        sources[record.get("source", "unknown")] += 1
        licenses[record.get("license", "unknown")] += 1
        if "split" in record:
            splits[record["split"]] += 1
        langs[detect_language(text)] += 1

    return {
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_words": n_words,
        "chars_per_doc": _percentiles(char_lengths),
        "words_per_doc": _percentiles(word_counts),
        "sources": dict(sources.most_common()),
        "splits": dict(splits),
        "languages": dict(langs.most_common()),
        "licenses": dict(licenses.most_common()),
    }


def write_report(stats: dict, out_path: Path) -> Path:
    """Write a machine-readable ``stats.json`` and a human ``REPORT.md``."""
    out_path.mkdir(parents=True, exist_ok=True)
    stats_json = out_path / "stats.json"
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Dataset Statistics Report", ""]
    for key in ("n_docs", "n_chars", "n_words"):
        lines.append(f"- **{key}**: {stats.get(key, 0):,}")
    lines.append("")
    lines.append("## Per-document length (chars)")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in stats.get("chars_per_doc", {}).items():
        lines.append(f"| {k} | {v:,.1f} |")
    lines.append("")
    lines.append("## Per-document length (words)")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in stats.get("words_per_doc", {}).items():
        lines.append(f"| {k} | {v:,.1f} |")
    lines.append("")
    if stats.get("languages"):
        lines.append("## Language distribution")
        lines.append("| language | docs |")
        lines.append("| --- | --- |")
        for k, v in stats.get("languages", {}).items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    if stats.get("sources"):
        lines.append("## Sources")
        lines.append("| source | docs |")
        lines.append("| --- | --- |")
        for k, v in stats.get("sources", {}).items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    if stats.get("licenses"):
        lines.append("## Licenses")
        lines.append("| license | docs |")
        lines.append("| --- | --- |")
        for k, v in stats.get("licenses", {}).items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    if stats.get("splits"):
        lines.append("## Split sizes")
        lines.append("| split | docs |")
        lines.append("| --- | --- |")
        for k, v in stats.get("splits", {}).items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    if stats.get("quality_scores"):
        quality_scores = stats["quality_scores"]
        lines.append("## Quality score distribution")
        for key in ("count", "min", "p10", "median", "mean", "p90", "max"):
            if key in quality_scores:
                value = quality_scores[key]
                if key == "count":
                    lines.append(f"- **{key}**: {value:,}")
                else:
                    lines.append(f"- **{key}**: {value:.3f}")
        histogram = quality_scores.get("histogram", {})
        if histogram:
            lines.extend(["", "| score range | records |", "| --- | ---: |"])
            lines.extend(f"| {bucket} | {count:,} |" for bucket, count in histogram.items())
        lines.append("")
    if stats.get("dedup"):
        dedup_stats = stats["dedup"]
        lines.append("## Deduplication")
        lines.append(f"- **input**: {dedup_stats['input']:,}")
        lines.append(f"- **removed exact**: {dedup_stats['removed_exact']:,}")
        lines.append(f"- **removed near**: {dedup_stats['removed_near']:,}")
        lines.append(f"- **dedup rate**: {dedup_stats['rate']:.2%}")
        lines.append("")

    report_path = out_path / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

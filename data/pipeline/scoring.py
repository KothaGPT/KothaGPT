"""Deterministic, model-free quality scores for instruction records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from data.pipeline.quality import detect_language

__all__ = ["score_distribution", "score_instruction_record"]

_TOKEN_RE = re.compile(r"[\w\u0980-\u09ff]+", re.UNICODE)
_SCORE_COMPONENTS = ("language", "diversity", "length", "provenance")


def _record_text(record: dict[str, Any]) -> str:
    return "\n".join(message["content"] for message in record["messages"])


def score_instruction_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable 0..1 score and its component scores."""
    messages = record["messages"]
    text = _record_text(record)
    assistant_text = "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    detected_language = detect_language(text)
    declared_language = record.get("language", "unknown")
    language = detected_language if declared_language == "unknown" else declared_language
    language_score = 1.0 if detected_language in {"bn", "en", "mixed"} else 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    diversity_score = len(set(tokens)) / len(tokens) if tokens else 0.0
    char_count = len(text)
    length_score = min(1.0, char_count / 40, 4000 / char_count) if char_count else 0.0
    provenance_score = float(bool(record.get("source") and record.get("license")))
    components = {
        "language": language_score,
        "diversity": diversity_score,
        "length": length_score,
        "provenance": provenance_score,
    }
    score = sum(components.values()) / len(_SCORE_COMPONENTS)
    return {
        "quality_score": round(score, 6),
        "quality_components": {key: round(value, 6) for key, value in components.items()},
        "language": record.get("language", language),
        "answer_chars": len(assistant_text),
    }


def score_distribution(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize score percentiles and fixed-width histogram bins."""
    scores = [float(record["quality_score"]) for record in records]
    if not scores:
        return {"count": 0, "histogram": {}}
    ordered = sorted(scores)
    bins: Counter[str] = Counter()
    for score in scores:
        lower = min(int(score * 10), 9) / 10
        bins[f"{lower:.1f}-{lower + 0.1:.1f}"] += 1
    return {
        "count": len(scores),
        "min": min(scores),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "median": ordered[int(0.50 * (len(ordered) - 1))],
        "mean": sum(scores) / len(scores),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "max": max(scores),
        "histogram": dict(sorted(bins.items())),
    }

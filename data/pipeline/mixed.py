"""Bangla-English (code-switched) text handling (Phase 2 — WS-4).

Provides script detection and light segmentation for mixed Bangla/Latin text.
The default tokenizer already keeps Latin words intact via its GPT-2-style
``▁`` marker; these helpers are used by the data pipeline and QA/RAG tooling to
split script runs and keep romanized Bengali words from being split mid-word.
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = [
    "bengali_ratio",
    "ensure_script_spacing",
    "is_bangla",
    "script_of",
    "segment_mixed",
    "split_script_runs",
]

Script = Literal["bn", "la", "other"]

_BENGALI_RE = "[\u0980-\u09ff]"
_LATIN_RE = r"[A-Za-z]"

_MIXED_BOUNDARY_RE = re.compile(rf"(?<=[A-Za-z])(?={_BENGALI_RE})|(?<={_BENGALI_RE})(?=[A-Za-z])")
_BENGALI_CHAR_RE = re.compile(_BENGALI_RE)
_LATIN_CHAR_RE = re.compile(_LATIN_RE)


def script_of(ch: str) -> Script:
    """Return the script bucket of a single character."""
    if _BENGALI_CHAR_RE.match(ch):
        return "bn"
    if _LATIN_CHAR_RE.match(ch):
        return "la"
    return "other"


def bengali_ratio(text: str) -> float:
    """Fraction of letters in ``text`` that are Bengali."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _BENGALI_CHAR_RE.match(c)) / len(letters)


def is_bangla(text: str, threshold: float = 0.5) -> bool:
    """True when at least ``threshold`` of the letters are Bengali."""
    return bengali_ratio(text) >= threshold


def split_script_runs(text: str) -> list[str]:
    """Split text into maximal runs of the same script.

    Example: ``"Sundayসকালে"`` -> ``["Sunday", "সকালে"]``.
    """
    runs: list[str] = []
    current = ""
    current_script: Script | None = None
    for ch in text:
        script = script_of(ch)
        if script == "other":
            script = current_script or "other"
        if script != current_script:
            if current:
                runs.append(current)
            current = ch
            current_script = script
        else:
            current += ch
    if current:
        runs.append(current)
    return runs


def segment_mixed(text: str) -> list[dict]:
    """Segment text into script-labelled spans.

    Returns a list of ``{"script": "bn"|"la"|"other", "text": ...}`` spans.
    """
    segments: list[dict] = []
    for run in split_script_runs(text):
        script = script_of(run[0])
        segments.append({"script": script, "text": run})
    return segments


def ensure_script_spacing(text: str) -> str:
    """Insert a single space between adjacent Latin and Bangla letter runs.

    Handles merged code-switched words such as ``"Sundayসকালে"`` (no space
    between scripts) so downstream tokenization does not join them.
    """
    return _MIXED_BOUNDARY_RE.sub(" ", text)

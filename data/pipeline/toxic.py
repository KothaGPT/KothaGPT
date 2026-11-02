"""Toxic-content filtering: slur/hate blocklists with an optional classifier.

A curated, word-boundary-matched blocklist of clearly harmful terms (Bangla +
English) is applied on every document by default; a hit rejects the document
with a rule id. An optional toxicity classifier callable can be supplied for
broader coverage — it rejects documents whose score crosses a threshold.

Disable with ``--no-toxic-check`` for research datasets that intentionally
contain toxic content.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable

__all__ = ["TOXIC_TERMS", "load_classifier", "toxic_gate", "toxic_hits"]

# category -> terms. Word-boundary matched on lowercased text. This is a
# starter list; expand from real corpus samples as they surface.
TOXIC_TERMS: dict[str, tuple[str, ...]] = {
    "slur": (
        "chink",
        "coon",
        "cunt",
        "dyke",
        "fag",
        "faggot",
        "gook",
        "kike",
        "nigger",
        "spic",
        "tranny",
        "wetback",
        "মাগী",
        "নিগ্রো",
        "বেশ্যা",
    ),
    "hate": (
        "heil hitler",
        "jews will not replace us",
        "white power",
        "কাফের",
        "ধর্মীয় বিদ্বেষ",
    ),
    "profanity": (
        "asshole",
        "bitch",
        "bullshit",
        "cock",
        "dickhead",
        "fuck",
        "fucking",
        "motherfucker",
        "shit",
        "slut",
        "whore",
        "চুদা",
        "মাদারচোদ",
        "মাদারচোদী",
    ),
}

Classifier = Callable[[str], float]

# Compile once: category -> list of (term, word-boundary pattern). Python's
# `\b` is unreliable for Bangla because vowel signs/virama (Mn) are not `\w`,
# so boundaries use lookarounds over word chars + the Bangla block (0980-09FF).
_BOUNDARY = r"(?<![\w\u0980-\u09ff])"
_WORD_CHAR = r"(?![\w\u0980-\u09ff])"
_COMPILED: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    cat: [(term, re.compile(_BOUNDARY + re.escape(term) + _WORD_CHAR)) for term in terms]
    for cat, terms in TOXIC_TERMS.items()
}


def toxic_hits(text: str) -> list[str]:
    """Return ``[category:term, ...]`` for every blocklisted term found."""
    lowered = text.lower()
    hits: list[str] = []
    for cat, patterns in _COMPILED.items():
        for term, pattern in patterns:
            if pattern.search(lowered):
                hits.append(f"{cat}:{term}")
    return hits


def load_classifier(path: str) -> Classifier:
    """Import a ``module:callable`` classifier from a string path.

    The callable must accept a string and return a toxicity score in ``[0,1]``.
    """
    if ":" not in path:
        raise ValueError(f"toxic classifier must be 'module:callable', got {path!r}")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    classifier = getattr(module, func_name)
    if not callable(classifier):
        raise TypeError(f"{path!r} does not resolve to a callable")
    return classifier


def toxic_gate(
    text: str,
    *,
    classifier: Classifier | None = None,
    classifier_threshold: float = 0.8,
) -> tuple[bool, list[str]]:
    """Return ``(keep, reasons)`` for the toxic-content gate.

    A blocklist hit always rejects. When ``classifier`` is provided, documents
    with ``classifier(text) >= classifier_threshold`` are also rejected.
    """
    reasons: list[str] = []

    hits = toxic_hits(text)
    if hits:
        reasons.append("toxic:" + ",".join(hits[:5]))
        return False, reasons

    if classifier is not None:
        score = float(classifier(text))
        if score >= classifier_threshold:
            reasons.append(f"toxic_classifier: {score:.3f}")
            return False, reasons

    return True, reasons

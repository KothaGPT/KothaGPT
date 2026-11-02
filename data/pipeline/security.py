"""WS-11 Dataset poisoning — label-flip / backdoor heuristics."""

from __future__ import annotations

import re

_TRIGGER_PATTERNS = [re.compile(r"\bbackdoor\b", re.IGNORECASE), re.compile(r"@@trigger@@")]

def is_poisoned(text: str, label: str | None = None) -> tuple[bool, str | None]:
    for rx in _TRIGGER_PATTERNS:
        if rx.search(text):
            return True, f"trigger:{rx.pattern}"
    # Label-flip: very short text with high-confidence label mismatch (heuristic)
    if label and len(text.split()) < 3 and label.lower() in {"positive", "negative"}:
        return True, "short_label_flip"
    return False, None

def scan_dataset(records: list[dict]) -> list[dict]:
    poisoned = []
    for r in records:
        text = r.get("text") or r.get("instruction") or ""
        label = r.get("label")
        flag, reason = is_poisoned(text, label)
        if flag:
            poisoned.append({"record": r, "reason": reason})
    return poisoned

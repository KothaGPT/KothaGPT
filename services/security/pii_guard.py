"""WS-8 Runtime PII guard — output masking."""

from __future__ import annotations

import re

_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d[\d\s\-\(\)]{8,}\d"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

def detect_pii(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for kind, rx in _PII_PATTERNS.items():
        found = rx.findall(text)
        if found:
            hits[kind] = found
    return hits

def mask_pii(text: str, policy: str = "mask") -> str:
    if policy == "drop" and detect_pii(text):
        return "[REDACTED PII]"
    out = text
    for kind, rx in _PII_PATTERNS.items():
        out = rx.sub(f"[{kind.upper()}_REDACTED]", out)
    return out

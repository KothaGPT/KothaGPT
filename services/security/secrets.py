"""WS-4 Secret isolation — vault-style with redaction."""

from __future__ import annotations

import os
import re

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9_\-]{8,})"),
    re.compile(r"(?i)(secret\s*[:=]\s*)([A-Za-z0-9_\-]{8,})"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),
]

class SecretsManager:
    def __init__(self):
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str, *, version: int = 1) -> None:
        self._store[f"{key}:v{version}"] = value
        self._store[key] = value

    def get(self, key: str) -> str | None:
        # Only via reference, not direct env leak
        return self._store.get(key) or os.getenv(key)

    def rotate(self, key: str, new_value: str) -> int:
        # Find current version
        versions = [int(k.split(":v")[1]) for k in self._store if k.startswith(f"{key}:v")]
        nxt = max(versions, default=0) + 1
        self.set(key, new_value, version=nxt)
        return nxt

def redact(text: str) -> str:
    out = text
    for rx in _SECRET_PATTERNS:
        out = rx.sub(lambda m: m.group(0).replace(m.group(2) if len(m.groups())>=2 else m.group(0), "***"), out)
    # generic: replace long tokens
    out = re.sub(r"[A-Za-z0-9_\-]{32,}", "***", out)
    return out

def scan_for_secrets(text: str) -> list[str]:
    hits = []
    for rx in _SECRET_PATTERNS:
        if rx.search(text):
            hits.append(rx.pattern)
    return hits

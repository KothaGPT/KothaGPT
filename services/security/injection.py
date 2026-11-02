"""WS-1 Prompt injection protection — input classification + template hardening."""

from __future__ import annotations

import re

# Known jailbreak / hidden-instruction patterns (Bangla + English)
_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"ignore\s+all\s+instructions",
    r"disregard\s+.*\s+instructions",
    r"you\s+are\s+now\s+.*DAN",
    r"system\s*:\s*",
    r"\[system\]",
    r"assistant\s*to\s*=\s*user",
    r"নির্দেশ\s+অগ্রাহ্য\s+কর",  # Bangla: ignore instructions
    r"পূর্ববর্তী\s+নির্দেশ",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

def is_injection(text: str) -> tuple[bool, str | None]:
    """Return (is_injection, matched_pattern) for input classification."""
    for pat, rx in zip(_INJECTION_PATTERNS, _COMPILED):
        if rx.search(text):
            return True, pat
    # Hidden instruction: excessive imperative verbs + role claims
    if re.search(r"(?:^|\n)\s*(?:User|System|Assistant)\s*:", text, re.IGNORECASE):
        return True, "role_claim"
    return False, None

def sanitize_context(text: str) -> str:
    """Harden RAG/tool content by quoting as data, never instructions."""
    # Wrap retrieved chunks as quoted data block
    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    return f"<data>\n{escaped}\n</data>"

def check_output(output: str) -> tuple[bool, str | None]:
    """Policy check: does output leak tools/roles or repeat injected instructions?"""
    if re.search(r"<(?:system|tool)>", output, re.IGNORECASE):
        return True, "leak_tool_role"
    return False, None


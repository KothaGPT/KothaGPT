"""Quality filters: language detection, PII scanning, length limits.

All rules are dependency-free. Language detection is script-based with a
Bangla word-confidence boost, tuned for the Bengali (Bangla) language and its
digraph-free Unicode block (U+0980–U+09FF).
"""

from __future__ import annotations

import re

__all__ = [
    "PII_MASK",
    "bengali_ratio",
    "contains_pii",
    "detect_language",
    "length_filter",
    "quality_filter",
    "redact_pii",
]

_BENGALI_START = 0x0980
_BENGALI_END = 0x0A00  # exclusive

_BANGLA_COMMON_WORDS = frozenset(
    {
        "আমি",
        "আমার",
        "আমরা",
        "তুমি",
        "তুমরা",
        "তোমার",
        "আপনি",
        "আপনারা",
        "আপনার",
        "করা",
        "করে",
        "করছে",
        "করবেন",
        "করতে",
        "হয়",
        "হতে",
        "হবে",
        "হয়েছে",
        "হচ্ছে",
        "এই",
        "ওই",
        "সেই",
        "যে",
        "ও",
        "আর",
        "না",
        "কিন্তু",
        "অথবা",
        "এবং",
        "যদি",
        "থেকে",
        "দিয়ে",
        "জন্য",
        "সাথে",
        "পরে",
        "আগে",
        "মধ্যে",
        "পাশে",
        "পর্যন্ত",
        "সব",
        "কিছু",
        "কোনো",
        "অনেক",
        "একটি",
        "একটা",
        "আছে",
        "ছিল",
        "গেছে",
        "নেই",
        "বাংলা",
        "বাংলাদেশ",
        "ঢাকা",
        "বছর",
        "মানুষ",
        "দেশ",
        "ভাষা",
        "লোক",
        "সময়",
        "কাজ",
        "কথা",
        "মত",
        "বলে",
        "চলবে",
        "দিন",
        "রাত",
        "জল",
        "বাড়ি",
        "ঘর",
    }
)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?88)?01[3-9]\d{8}"
    r"|\+?880\s?1[3-9]\d{8}"
    r"|\b0\d{2,3}[-.\s]?\d{4}[-.\s]?\d{4}\b"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# Bangladeshi national ID: 10 or 17 digits.
_NID_RE = re.compile(r"(?<!\d)(?:\d{17}|\d{10})(?!\d)")
# Passport (e.g. MRP "A0123456"): 1-2 letters followed by 6-8 digits.
_PASSPORT_RE = re.compile(r"\b[A-Z]{1,2}\d{6,8}\b")
# Street addresses with a number prefix (English + Bangla street markers).
_ADDRESS_EN_RE = re.compile(
    r"\b\d{1,5}\s+(road|street|st\.?|avenue|ave\.?|lane|lane|bd|society|sector)\b",
    re.IGNORECASE,
)
_ADDRESS_BN_RE = re.compile(r"(?:হাউস|বাসা|বাড়ি|রোড|সড়ক|গলি|লেন)\s*[০-৯0-9][০-৯0-9\-/]*")

PII_TYPES = ("email", "url", "phone", "ip", "credit_card", "nid", "passport", "address")

_PII_RULES = (
    ("email", _EMAIL_RE),
    ("url", _URL_RE),
    ("phone", _PHONE_RE),
    ("ip", _IPV4_RE),
    ("ip", _IPV6_RE),
    ("credit_card", _CARD_RE),
    ("nid", _NID_RE),
    ("passport", _PASSPORT_RE),
    ("address", _ADDRESS_EN_RE),
    ("address", _ADDRESS_BN_RE),
)


def bengali_ratio(text: str) -> float:
    """Fraction of alphabetic characters that belong to the Bengali block."""
    bengali = 0
    letters = 0
    for ch in text:
        cp = ord(ch)
        if _BENGALI_START <= cp < _BENGALI_END:
            bengali += 1
            letters += 1
        elif ch.isalpha():
            letters += 1
    return bengali / letters if letters else 0.0


def _latin_ratio(text: str) -> float:
    letters = 0
    latin = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if "a" <= ch.lower() <= "z":
                latin += 1
    return latin / letters if letters else 0.0


def _word_boost(text: str) -> float:
    """Fraction of space-separated words that are common Bangla words."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _BANGLA_COMMON_WORDS)
    return hits / len(words)


def detect_language(text: str) -> str:
    """Classify text as ``bn``, ``en``, ``mixed`` or ``other``.

    Bengali script presence is the primary signal; a common-word boost lifts
    short Bangla strings written without diacritics, and Latin ratio resolves
    English-only text.
    """
    ratio = bengali_ratio(text)
    latin = _latin_ratio(text)
    boost = _word_boost(text)
    if ratio >= 0.8:
        return "bn"
    if ratio > 0.0 and latin >= 0.15:
        return "mixed"
    if ratio >= 0.4 and boost >= 0.05:
        return "bn"
    if ratio > 0.0:
        return "bn"
    if latin >= 0.5 and boost == 0.0:
        return "en"
    return "other"


def contains_pii(text: str) -> list[str]:
    """Return the list of PII types found in ``text`` (empty when clean)."""
    found: list[str] = []
    for label, pattern in _PII_RULES:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


PII_MASK = "[PII]"


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace every PII span with :data:`PII_MASK`; ``(redacted, counts)``.

    ``counts`` maps PII type -> number of masked spans. The caller can then
    keep the redacted document instead of dropping it.
    """
    redacted = text
    counts: dict[str, int] = {}
    for label, pattern in _PII_RULES:
        matches = list(pattern.finditer(redacted))
        if not matches:
            continue
        counts[label] = len(matches)
        for match in reversed(matches):
            start, end = match.span()
            redacted = redacted[:start] + PII_MASK + redacted[end:]
    return redacted, counts


def length_filter(
    text: str, *, min_chars: int, max_chars: int, min_words: int = 20
) -> tuple[bool, list[str]]:
    """Reject texts that are too short, too long or nearly empty."""
    reasons: list[str] = []
    if len(text) < min_chars:
        reasons.append(f"too_short: {len(text)} chars")
    if len(text) > max_chars:
        reasons.append(f"too_long: {len(text)} chars")
    words = text.split()
    if len(words) < min_words:
        reasons.append(f"too_few_words: {len(words)}")
    return (not reasons, reasons)


def quality_filter(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_words: int,
    require_bangla: bool,
    min_bangla_ratio: float,
    allow_pii: bool,
) -> tuple[bool, list[str]]:
    """Run the full quality gate; returns ``(keep, reasons)``."""
    reasons: list[str] = []
    if require_bangla:
        lang = detect_language(text)
        if lang not in ("bn", "mixed"):
            reasons.append(f"language: {lang}")
        elif lang == "mixed" and bengali_ratio(text) < min_bangla_ratio:
            reasons.append(f"low_bangla_ratio: {bengali_ratio(text):.3f}")
    if not allow_pii:
        pii = contains_pii(text)
        if pii:
            reasons.append(f"pii: {','.join(pii)}")
    _, len_reasons = length_filter(
        text, min_chars=min_chars, max_chars=max_chars, min_words=min_words
    )
    reasons.extend(len_reasons)
    return (not reasons, reasons)

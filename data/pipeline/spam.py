"""Spam filtering: domain/phrase blocklists and repetition heuristics.

Dependency-free heuristics that assign every document a spam score in ``[0,1]``.
A document is rejected when ``score >= threshold`` (default 0.6). Signals:

- presence of a known spam domain (definitive, +1.0)
- promo/solicitation phrases (Bangla + English, +0.35 each)
- token repetition (a single token dominating the doc)
- long character bursts (e.g. ``!!!!`` / ``?????????``)
- excessive ALLCAPS or a high URL density (link farms)

Tune with ``--spam-threshold`` or disable with ``--no-spam-check``.
"""

from __future__ import annotations

import re

__all__ = ["SPAM_DOMAINS", "SPAM_PHRASES", "spam_gate", "spam_score"]

# Representative blocklist; extend via the blocklist file or by adding entries.
SPAM_DOMAINS: frozenset[str] = frozenset(
    {
        "1xbet.com",
        "10bet.com",
        "bet365.com",
        "casino.com",
        "casinoguru.com",
        "easyloans.com",
        "instant-loans.com",
        "lotterywin.com",
        "partycasino.com",
        "prizefinder.com",
        "quickcash.com",
        "royalclub.com",
    }
)

# Promo/solicitation phrases. Word-boundary matched; two hits (or one plus any
# other signal) are usually enough to exceed the default threshold.
SPAM_PHRASES: tuple[str, ...] = (
    "buy now",
    "cash prize",
    "casino",
    "click here",
    "earn money",
    "free gift",
    "free recharge",
    "limited time",
    "lottery",
    "win prize",
    "আয় করুন",
    "ইনকাম",
    "ক্লিক করুন",
    "ক্যাসিনো",
    "দ্রুত টাকা",
    "ফ্রি রিচার্জ",
    "বাজি ধরুন",
    "ভাগ্য পরীক্ষা",
    "পুরস্কার জিতুন",
    "ঋণ দিন",
    "লটারি",
    "হ্যাক",
)

_DOMAIN_RE = re.compile(r"\b([\w-]+\.(?:com|net|org|info|biz|xyz|club|online))\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_EXCLAIM_RE = re.compile(r"[!]{3,}")
_ALLCAPS_RE = re.compile(r"[A-Z]{4,}")


def _domain_hits(text: str) -> list[str]:
    hits = []
    for match in _DOMAIN_RE.findall(text):
        domain = match.lower()
        if domain in SPAM_DOMAINS or any(domain.endswith("." + d) for d in SPAM_DOMAINS):
            hits.append(domain)
    return hits


def _phrase_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [p for p in SPAM_PHRASES if re.search(r"\b" + re.escape(p) + r"\b", lowered)]


def _repetition_score(text: str) -> float:
    """Score the dominance of the most frequent token (0.0 -> 0.6).

    Only applies to documents with at least 10 tokens so short, legitimately
    repetitive prose (headlines, captions) is not penalised.
    """
    tokens = re.findall(r"[\w\u0980-\u09ff]+", text.lower())
    if len(tokens) < 10:
        return 0.0
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    peak = max(counts.values())
    ratio = peak / len(tokens)
    if ratio >= 0.5:
        return 0.6
    if ratio >= 0.3:
        return 0.4
    return 0.0


def _burst_score(text: str) -> float:
    """Long runs of a single character (!!!!, ......, ________) push up score."""
    longest = 0
    run = 0
    prev = ""
    for ch in text:
        if ch == prev:
            run += 1
        else:
            run = 1
        longest = max(longest, run)
        prev = ch
    if longest >= 10:
        return 0.4
    if longest >= 6:
        return 0.2
    return 0.0


def _caps_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if len(w) >= 4 and w.isupper())
    return 0.2 if caps / len(words) >= 0.5 else 0.0


def _url_density_score(text: str) -> float:
    urls = len(_URL_RE.findall(text))
    words = len(text.split())
    if words == 0:
        return 0.0
    return 0.3 if urls / words > 0.02 else 0.0


def spam_score(text: str) -> tuple[float, list[str]]:
    """Return ``(score, reasons)`` for ``text``; score is capped at 1.0."""
    score = 0.0
    reasons: list[str] = []

    domains = _domain_hits(text)
    if domains:
        score += 1.0
        reasons.append(f"spam_domain: {','.join(domains[:3])}")

    phrases = _phrase_hits(text)
    if phrases:
        score += min(1.0, 0.35 * len(phrases))
        reasons.append(f"spam_phrase: {','.join(phrases[:3])}")

    rep = _repetition_score(text)
    if rep:
        score += rep
        reasons.append(f"repetition: {rep:.1f}")

    burst = _burst_score(text)
    if burst:
        score += burst
        reasons.append(f"char_burst: {burst:.1f}")

    caps = _caps_score(text)
    if caps:
        score += caps
        reasons.append("allcaps")

    density = _url_density_score(text)
    if density:
        score += density
        reasons.append("url_density")

    return min(1.0, score), reasons


def spam_gate(text: str, threshold: float = 0.6) -> tuple[bool, list[str]]:
    """Return ``(keep, reasons)``; a doc is spam when ``score >= threshold``."""
    score, reasons = spam_score(text)
    if score >= threshold:
        reasons.insert(0, f"spam_score: {score:.2f}")
        return False, reasons
    return True, reasons

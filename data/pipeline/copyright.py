"""Copyright/licensing validation gate.

Every record should carry a machine-readable license so releases can prove
their provenance. This module normalizes license strings, checks them against
an allow-list of open licenses, resolves a per-source license map, and blocks
known-copyrighted titles.

Usage (pipeline)::

    python -m data.pipeline.cli run --require-license --license-map data/raw/.licenses.json

The license map is JSON keyed by source relative path (glob patterns and a
``"*"`` default supported), e.g.::

    {"bn_wikipedia/*": "CC-BY-SA-4.0", "samanantar.jsonl": "CC-BY-SA-4.0"}
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ALLOWLIST",
    "copyright_gate",
    "is_allowed",
    "load_license_map",
    "load_title_blocklist",
    "normalize_license",
    "resolve_license",
]

# Conservative default: permissive + attribution/open-data licenses only.
# Non-commercial (NC/ND) and copyleft-strong licenses are excluded unless the
# caller supplies an explicit allow-list.
DEFAULT_ALLOWLIST = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by",
        "cc-by-sa",
        "cc0",
        "isc",
        "mit",
        "odc-by",
        "odc-odbl",
        "public-domain",
        "unlicense",
    }
)

_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")
# Creative Commons "words" -> short rights codes.
_CC_WORD_MAP = {
    "attribution": "by",
    "sharealike": "sa",
    "noncommercial": "nc",
    "noderivatives": "nd",
    "zero": "",
}


def normalize_license(license_str: str) -> str:
    """Canonicalize a license string to a stable dotted id (``cc-by-sa-4.0``)."""
    s = license_str.strip().lower()
    s = s.replace("creative commons", "cc").replace("creativecommons", "cc")

    # Pull a dotted version (4.0, 2.0) out so it isn't split across tokens.
    version = ""
    match = _VERSION_RE.search(s)
    if match:
        version = match.group(0)
        s = s.replace(match.group(0), " ")

    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t]

    if "public" in tokens and "domain" in tokens:
        return "public-domain"

    if tokens and tokens[0] in ("cc", "cc0", "cc-zero"):
        parts = [_CC_WORD_MAP.get(t, t) for t in tokens[1:]]
        parts = [p for p in parts if p]
        base = "cc0" if not parts else "cc-" + "-".join(parts)
        return base + ("-" + version if version else "")

    base = "-".join(tokens)
    return base + ("-" + version if version else "")


def is_allowed(license_str: str, allowlist: frozenset[str] | None = None) -> bool:
    """Return whether ``license_str`` normalizes into ``allowlist``.

    Matching is version-insensitive: ``cc-by-sa-4.0`` satisfies the
    ``cc-by-sa`` allow-list entry, while ``cc-by-nc`` never does.
    """
    allowed = allowlist or DEFAULT_ALLOWLIST
    norm = normalize_license(license_str)
    if norm in allowed:
        return True
    base = _VERSION_RE.sub("", norm).rstrip("-")
    return bool(base) and base in allowed


def load_license_map(path: str | Path) -> dict[str, str]:
    """Load a JSON ``{source_pattern: license}`` map (empty when missing)."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{p}: expected a JSON object of {{source: license}}")
    return {str(k): str(v) for k, v in data.items()}


def load_title_blocklist(path: str | Path | None) -> frozenset[str]:
    """Load a one-title-per-line blocklist of known copyrighted works."""
    if not path:
        return frozenset()
    p = Path(path)
    if not p.exists():
        return frozenset()
    titles = frozenset(
        line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    return titles


def resolve_license(record: dict[str, Any], license_map: dict[str, str]) -> str | None:
    """Resolve a record's license from ``record['license']`` or its source.

    ``record['source']`` is matched against the map keys exactly, then as glob
    patterns, then a ``"*"`` default key.
    """
    explicit = record.get("license")
    if explicit:
        return str(explicit)
    source = record.get("source")
    if not source or not license_map:
        return None
    if source in license_map:
        return license_map[source]
    for pattern, license_val in license_map.items():
        if pattern != "*" and fnmatch(source, pattern):
            return license_val
    return license_map.get("*")


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^\w\u0980-\u09ff]+", title.lower()) if t)


def known_copyrighted_title(title: str, blocklist: frozenset[str]) -> bool:
    """Return whether ``title`` matches a blocklist title (normalized compare)."""
    if not title or not blocklist:
        return False
    if title in blocklist:
        return True
    tokens = _title_tokens(title)
    return any(tokens == _title_tokens(b) for b in blocklist)


def copyright_gate(
    record: dict[str, Any],
    *,
    license_map: dict[str, str] | None = None,
    allowlist: frozenset[str] | None = None,
    require_license: bool = False,
    title_blocklist: frozenset[str] = frozenset(),
) -> tuple[bool, list[str]]:
    """Return ``(keep, reasons)`` for the copyright/licensing gate.

    When ``require_license`` is set, records without a resolvable, allowed
    license are rejected. Records that resolve to a license via the map get
    that license attached so downstream stats/provenance can report it.
    """
    reasons: list[str] = []
    license_val = resolve_license(record, license_map or {})
    if license_val:
        record.setdefault("license", license_val)
    if require_license:
        if not license_val:
            reasons.append("license: missing")
        elif not is_allowed(license_val, allowlist):
            reasons.append(f"license: {license_val!r} not allowed")

    title = record.get("title")
    if known_copyrighted_title(title, title_blocklist):
        reasons.append(f"title: known copyrighted work: {title!r}")

    return (not reasons, reasons)

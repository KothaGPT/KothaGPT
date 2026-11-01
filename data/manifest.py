"""KGPT-003 — Dataset provenance and license registry.

Every corpus source used by the pipeline must be registered here with its
license, provenance metadata, and a content checksum. The registry is the
single source of truth for "is this source legally acceptable for training?".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DatasetRegistry",
    "DatasetSource",
    "LicenseAllowlist",
    "ManifestEntry",
    "get_registry",
    "load_manifest",
    "load_registry",
    "register_manifest",
    "save_registry",
]


_DEFAULT_ALLOWLIST = frozenset(
    {
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
        "cc0-1.0",
        "mit",
        "apache-2.0",
        "bsd-3-clause",
        "bsd-2-clause",
        "isc",
        "mpl-2.0",
        "cc-by-nc-nd-4.0",
        "odc-by",
    }
)

_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")


@dataclass(frozen=True, slots=True)
class LicenseAllowlist:
    allowed: frozenset[str] = _DEFAULT_ALLOWLIST
    blocked: frozenset[str] = frozenset()

    def is_allowed(self, license_id: str) -> bool:
        normalized = (license_id or "").strip().lower()
        if not normalized:
            return False
        if normalized in self.blocked:
            return False
        # Version-insensitive matching: "odc-by-1.0" satisfies "odc-by".
        base = _VERSION_RE.sub("", normalized).rstrip("-")
        return (
            normalized in self.allowed
            or bool(base) and base in self.allowed
            or "*" in self.allowed
        )


@dataclass
class DatasetSource:
    name: str
    url: str
    license: str
    license_url: str | None = None
    provenance: str = ""
    collection_date: str = ""
    checksum: str = ""
    content_type: str = "text"
    notes: str = ""
    allow_training: bool = True
    allow_distribution: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source name is required")
        if not self.license:
            raise ValueError("source license is required")
        if not self.collection_date:
            self.collection_date = datetime.now(UTC).isoformat(timespec="seconds")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """A single validated entry in ``corpora.manifest.json``.

    ``license``/``license_url``/``provenance`` are mandatory so every ingested
    source carries verifiable provenance metadata (KGPT-003 / issue #8).
    """

    name: str
    url: str
    license: str
    license_url: str
    provenance: str
    content_type: str = "text"
    notes: str = ""
    allow_training: bool = True
    allow_distribution: bool = True
    collection_date: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("manifest entry missing 'name'")
        if not self.url:
            raise ValueError(f"manifest entry {self.name!r} missing 'url'")
        if not self.license:
            raise ValueError(f"manifest entry {self.name!r} missing 'license'")
        if not self.license_url:
            raise ValueError(f"manifest entry {self.name!r} missing 'license_url'")
        if not self.provenance:
            raise ValueError(f"manifest entry {self.name!r} missing 'provenance'")
        if not self.collection_date:
            object.__setattr__(
                self, "collection_date", datetime.now(UTC).isoformat(timespec="seconds")
            )

    def to_source(self) -> DatasetSource:
        return DatasetSource(
            name=self.name,
            url=self.url,
            license=self.license,
            license_url=self.license_url,
            provenance=self.provenance,
            collection_date=self.collection_date,
            content_type=self.content_type,
            notes=self.notes,
            allow_training=self.allow_training,
            allow_distribution=self.allow_distribution,
        )


class DatasetRegistry:
    """In-memory registry of corpus sources with optional JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._sources: dict[str, DatasetSource] = {}
        self._allowlist = LicenseAllowlist()
        if self.path and self.path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for raw in data.get("sources", []):
            src = DatasetSource(**raw)
            self._sources[src.name] = src
        allowed = data.get("allowlist")
        if allowed is not None:
            self._allowlist = LicenseAllowlist(allowed=frozenset(allowed))

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path else (self.path or Path("data/processed/registry.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sources": [s.as_dict() for s in self._sources.values()],
            "allowlist": sorted(self._allowlist.allowed),
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return out

    def register(self, source: DatasetSource) -> DatasetSource:
        if source.name in self._sources:
            raise ValueError(f"source already registered: {source.name}")
        self._sources[source.name] = source
        return source

    def get(self, name: str) -> DatasetSource | None:
        return self._sources.get(name)

    def list(self) -> list[DatasetSource]:
        return list(self._sources.values())

    def set_allowlist(self, allowlist: LicenseAllowlist) -> None:
        self._allowlist = allowlist

    @property
    def allowlist(self) -> LicenseAllowlist:
        return self._allowlist

    def is_allowed(self, name: str) -> tuple[bool, str]:
        """Return (allowed, reason) for a registered source."""
        src = self._sources.get(name)
        if src is None:
            return False, f"source {name!r} is not registered"
        if not self._allowlist.is_allowed(src.license):
            return False, f"license {src.license!r} for {name!r} is not in the allowlist"
        if not src.allow_training:
            return False, f"source {name!r} opts out of training use"
        return True, "ok"

    def enforce(self, names: list[str]) -> None:
        """Raise ``ValueError`` if any source is not allowed."""
        for name in names:
            ok, reason = self.is_allowed(name)
            if not ok:
                raise ValueError(f"dataset source rejected: {reason}")


_MANIFEST_REQUIRED = ("name", "url", "license", "license_url", "provenance")


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load and validate a corpora manifest JSON file.

    Returns a list of :class:`ManifestEntry`. Raises ``ValueError`` when the
    manifest is missing, malformed, or any entry lacks required provenance /
    license metadata.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("datasets", [])
    if not isinstance(data, list):
        raise TypeError(f"{p}: manifest must be a JSON list (or {{datasets: [...]}})")
    entries: list[ManifestEntry] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise TypeError(f"{p}: entry #{index} is not an object")
        missing = [k for k in _MANIFEST_REQUIRED if k not in raw or not raw[k]]
        if missing:
            raise ValueError(
                f"{p}: entry #{index} missing required fields: {missing}"
            )
        entry = ManifestEntry(
            name=str(raw["name"]),
            url=str(raw["url"]),
            license=str(raw["license"]),
            license_url=str(raw["license_url"]),
            provenance=str(raw["provenance"]),
            content_type=str(raw.get("content_type", "text")),
            notes=str(raw.get("notes", "")),
            allow_training=bool(raw.get("allow_training", True)),
            allow_distribution=bool(raw.get("allow_distribution", True)),
            collection_date=str(raw.get("collection_date", "")),
        )
        entries.append(entry)
    return entries


def register_manifest(
    registry: DatasetRegistry,
    manifest_path: str | Path,
    *,
    strict: bool = True,
) -> dict:
    """Register every manifest source into ``registry``.

    Returns a report dict with ``registered``, ``rejected`` counts and per-entry
    reasons. Validation is atomic: in strict (production) mode, if any entry is
    invalid (disallowed license or duplicate name) the whole registration raises
    and nothing is registered. In development bypass mode (``strict=False``),
    valid entries are registered and invalid ones are skipped and reported.

    The report is reproducible: the same manifest + registry state yields the
    same result.
    """
    entries = load_manifest(manifest_path)
    registered: list[str] = []
    rejected: list[dict[str, str]] = []
    valid: list[ManifestEntry] = []
    for entry in entries:
        # License allowlist check (independent of name registration).
        if not registry.allowlist.is_allowed(entry.license):
            rejected.append(
                {"name": entry.name, "reason": f"license {entry.license!r} not in allowlist"}
            )
            continue
        # Duplicate name check (against registry and against other manifest entries).
        if registry.get(entry.name) is not None or any(v.name == entry.name for v in valid):
            rejected.append(
                {"name": entry.name, "reason": f"source {entry.name!r} already registered"}
            )
            continue
        valid.append(entry)

    if strict and rejected:
        reasons = "; ".join(f"{r['name']}: {r['reason']}" for r in rejected)
        raise ValueError(f"manifest entry rejected: {reasons}")

    for entry in valid:
        registry.register(entry.to_source())
        registered.append(entry.name)

    return {
        "manifest": str(Path(manifest_path)),
        "total": len(entries),
        "registered": len(registered),
        "rejected": len(rejected),
        "registered_sources": registered,
        "rejected_entries": rejected,
    }


_REGISTRY: DatasetRegistry | None = None


def get_registry() -> DatasetRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DatasetRegistry(os.getenv("KOTHAGPT_DATASET_REGISTRY"))
    return _REGISTRY


def load_registry(path: str | Path) -> DatasetRegistry:
    return DatasetRegistry(path)


def save_registry(registry: DatasetRegistry, path: str | Path | None = None) -> Path:
    return registry.save(path)
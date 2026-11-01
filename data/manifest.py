"""KGPT-003 — Dataset provenance and license registry.

Every corpus source used by the pipeline must be registered here with its
license, provenance metadata, and a content checksum. The registry is the
single source of truth for "is this source legally acceptable for training?".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DatasetSource",
    "DatasetRegistry",
    "LicenseAllowlist",
    "get_registry",
    "load_registry",
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
    }
)


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
        return normalized in self.allowed or "*" in self.allowed


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
            self.collection_date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
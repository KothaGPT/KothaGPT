"""Dataset statistics and versioning metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

__all__ = ["DatasetVersion", "write_manifest"]


def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class DatasetVersion:
    """Metadata describing one emitted, immutable dataset snapshot."""

    version_id: str
    created_at: str
    config: dict
    counts: dict
    shards: list[dict]
    files: dict
    version_label: str | None = None

    @classmethod
    def create(
        cls,
        *,
        config: dict,
        counts: dict,
        shards: list[dict],
        files: dict,
        version_label: str | None = None,
    ) -> DatasetVersion:
        """Build a version, computing a content-addressed ``version_id``."""
        created_at = datetime.now(UTC).isoformat()
        base = {
            "created_at": created_at,
            "config": config,
            "counts": counts,
            "shards": shards,
            "files": files,
            "version_label": version_label,
        }
        digest = sha256(_canonical_json(base).encode("utf-8")).hexdigest()[:12]
        base["version_id"] = digest
        return cls(**base)

    def to_dict(self) -> dict:
        return asdict(self)


def write_manifest(version: DatasetVersion, version_dir: Path) -> Path:
    """Persist the version manifest and a ``CURRENT`` pointer file.

    ``version_dir/MANIFEST.json`` holds the full metadata; ``CURRENT`` at the
    parent of ``version_dir`` points at the latest version id.
    """
    version_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = version_dir / "MANIFEST.json"
    manifest_path.write_text(_canonical_json(version.to_dict()) + "\n", encoding="utf-8")
    current = version_dir.parent / "CURRENT"
    current.write_text(version.version_id + "\n", encoding="utf-8")
    return manifest_path

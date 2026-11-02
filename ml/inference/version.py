"""WS-10 Version management — immutable versions, stable alias, rollback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class VersionRecord:
    version: str
    digest: str
    artifact_path: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    superseded_by: str | None = None

class VersionManager:
    def __init__(self):
        self._versions: dict[str, list[VersionRecord]] = {}  # model_id -> versions
        self._stable: dict[str, str] = {}  # model_id -> version

    def _digest(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def register_version(self, model_id: str, artifact_path: str, metadata: dict[str, Any] | None = None) -> VersionRecord:
        meta = metadata or {}
        digest = self._digest({"artifact": artifact_path, "meta": meta})
        version = f"v{len(self._versions.get(model_id, []))+1}-{digest}"
        rec = VersionRecord(version=version, digest=digest, artifact_path=artifact_path)
        self._versions.setdefault(model_id, []).append(rec)
        # latest is last, stable is explicitly set or first
        if model_id not in self._stable:
            self._stable[model_id] = version
        return rec

    def list_versions(self, model_id: str) -> list[VersionRecord]:
        return list(self._versions.get(model_id, []))

    def get_stable(self, model_id: str) -> VersionRecord | None:
        ver = self._stable.get(model_id)
        if not ver:
            return None
        return next((v for v in self._versions.get(model_id, []) if v.version == ver), None)

    def promote_stable(self, model_id: str, version: str) -> VersionRecord:
        if version not in [v.version for v in self._versions.get(model_id, [])]:
            raise ValueError(f"version {version!r} not found for {model_id}")
        old = self._stable.get(model_id)
        if old:
            # mark superseded
            for v in self._versions[model_id]:
                if v.version == old:
                    v.superseded_by = version
        self._stable[model_id] = version
        return self.get_stable(model_id)  # type: ignore

    def rollback(self, model_id: str) -> VersionRecord | None:
        vers = self._versions.get(model_id, [])
        if len(vers) < 2:
            raise ValueError("no prior version to rollback to")
        # stable is last promoted, rollback to previous in history
        current = self._stable.get(model_id)
        # Find index of current
        idx = next((i for i, v in enumerate(vers) if v.version == current), len(vers)-1)
        prev = vers[max(0, idx-1)]
        return self.promote_stable(model_id, prev.version)

    def audit_log(self) -> list[dict[str, Any]]:
        return [{"model_id": mid, "stable": ver, "versions": [v.version for v in vs]} for mid, vs in self._versions.items() for ver in [self._stable.get(mid)]]

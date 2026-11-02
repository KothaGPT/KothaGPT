"""WS-9 Audit logs — append-only hash chain."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


class AuditLog:
    def __init__(self):
        self.entries: list[dict[str, Any]] = []
        self._prev_hash = "0"*64

    def log(self, actor: str, action: str, resource: str, decision: str, tenant: str = "default", trace_id: str = "") -> dict[str, Any]:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "decision": decision,
            "tenant": tenant,
            "trace_id": trace_id,
            "prev_hash": self._prev_hash,
        }
        payload = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = hashlib.sha256(payload).hexdigest()
        self._prev_hash = entry["hash"]
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = "0"*64
        for e in self.entries:
            h = e["hash"]
            # recompute without hash
            copy = {k: v for k, v in e.items() if k != "hash"}
            # But copy's prev_hash should match prev
            if copy["prev_hash"] != prev:
                return False
            if hashlib.sha256(json.dumps(copy, sort_keys=True).encode()).hexdigest() != h:
                return False
            prev = h
        return True

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        out = []
        for e in self.entries:
            if all(e.get(k) == v for k, v in filters.items()):
                out.append(e)
        return out

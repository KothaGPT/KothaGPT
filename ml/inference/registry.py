"""Model registry — Postgres-backed with in-memory fallback (WS-9)."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelRecord:
    id: str
    name: str
    family: str = "kothagpt"
    version: str = "0.2.0"
    quant_level: str = "none"
    device_profile: str = "cpu"
    artifact_path: str = ""
    digest: str = ""
    status: str = "ready"
    context_window: int = 8192
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("DATABASE_URL")
        self._mem: dict[str, ModelRecord] = {}
        self._init_defaults()

    def _init_defaults(self):
        tiers = [
            ModelRecord(id="kothagpt-nano", name="Kotha Nano", version="0.2.0",
                        family="kothagpt-nano", device_profile="cpu",
                        context_window=2048, quant_level="int8"),
            ModelRecord(id="kothagpt-small", name="Kotha Small", version="0.2.0",
                        family="kothagpt-small", device_profile="cpu",
                        context_window=4096, quant_level="none"),
            ModelRecord(id="kothagpt-medium", name="Kotha Medium", version="0.2.0",
                        family="kothagpt-medium", device_profile="gpu",
                        context_window=8192, quant_level="int8"),
            ModelRecord(id="kothagpt-large", name="Kotha Large", version="0.2.0",
                        family="kothagpt-large", device_profile="gpu",
                        context_window=16384, quant_level="int4"),
            ModelRecord(id="kothagpt", name="Kotha GPT Base", version="0.2.0",
                        family="kothagpt", device_profile="cpu",
                        context_window=4096, quant_level="none"),
            ModelRecord(id="kothagpt-embed", name="Kotha Embed", version="0.2.0",
                        family="kothagpt-embed", device_profile="cpu",
                        context_window=8192),
            ModelRecord(id="kothagpt-rerank", name="Kotha Rerank", version="0.2.0",
                        family="kothagpt-rerank", device_profile="cpu",
                        context_window=8192),
            ModelRecord(id="kothagpt-moe", name="Kotha MoE Frontier", version="0.2.0",
                        family="kothagpt-moe", device_profile="gpu",
                        context_window=16384, quant_level="int4", status="research"),
        ]
        for m in tiers:
            self._mem[m.id] = m

    def _ensure_table(self, engine) -> None:
        # Provision via migration SQL, not just metadata
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS model_registry (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                )
            )

    def list_models(self) -> list[ModelRecord]:
        if self.url:
            import sqlalchemy
            from sqlalchemy import MetaData, Table, Column, String, select, create_engine

            engine = create_engine(self.url)
            self._ensure_table(engine)
            meta = MetaData()
            tbl = Table(
                "model_registry",
                meta,
                Column("id", String, primary_key=True),
                Column("data", String),
                extend_existing=True,
            )
            with engine.connect() as conn:
                rows = conn.execute(select(tbl.c.data)).fetchall()
                if rows:
                    import json

                    merged = dict(self._mem)
                    for r in rows:
                        rec = ModelRecord(**json.loads(r[0]))
                        merged[rec.id] = rec
                    return list(merged.values())
        return list(self._mem.values())

    def get(self, model_id: str) -> ModelRecord | None:
        return next((m for m in self.list_models() if m.id == model_id), None)

    def register(self, rec: ModelRecord) -> ModelRecord:
        if self.url:
            import json
            from sqlalchemy import MetaData, Table, Column, String, create_engine, text

            engine = create_engine(self.url)
            self._ensure_table(engine)
            meta = MetaData()
            tbl = Table(
                "model_registry",
                meta,
                Column("id", String, primary_key=True),
                Column("data", String),
                extend_existing=True,
            )
            # Duplicate policy: upsert (replace) — applied to both DB and memory
            with engine.begin() as conn:
                # Use ON CONFLICT for Postgres, fallback to delete+insert for SQLite
                try:
                    conn.execute(
                        text(
                            "INSERT INTO model_registry (id, data) VALUES (:id, :data) "
                            "ON CONFLICT(id) DO UPDATE SET data = EXCLUDED.data"
                        ),
                        {"id": rec.id, "data": json.dumps(asdict(rec))},
                    )
                except Exception:
                    # Fallback for SQLite without ON CONFLICT support in older versions
                    conn.execute(text("DELETE FROM model_registry WHERE id = :id"), {"id": rec.id})
                    conn.execute(
                        text("INSERT INTO model_registry (id, data) VALUES (:id, :data)"),
                        {"id": rec.id, "data": json.dumps(asdict(rec))},
                    )
        self._mem[rec.id] = rec
        return rec


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

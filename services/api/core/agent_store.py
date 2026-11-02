"""Durable stores for agent/run state.

``MockBackend`` keeps agents and runs across requests, but on ephemeral hosts
(e.g. Vercel serverless functions) in-process state disappears with every
instance. Configure one of the following so state survives instance restarts:

- ``KOTHAGPT_AGENT_STORE_URL``: explicit override (redis:// or a SQLAlchemy URL)
- ``REDIS_URL``: Redis hash storage
- ``DATABASE_URL``: SQL storage via SQLAlchemy (PostgreSQL in production)

With none set, state stays in memory as before.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import Any

_AGENT_SCOPE_PREFIX = "kothagpt:"


class AgentStore(ABC):
    """Minimal hash semantics: named maps of string keys to JSON strings."""

    @abstractmethod
    def hset(self, scope: str, key: str, value: str) -> None: ...

    @abstractmethod
    def hget(self, scope: str, key: str) -> str | None: ...

    @abstractmethod
    def hgetall(self, scope: str) -> dict[str, str]: ...

    @abstractmethod
    def hdel(self, scope: str, key: str) -> None: ...

    def close(self) -> None:
        """Release underlying connections. No-op for stores without them."""


class MemoryStore(AgentStore):
    def __init__(self) -> None:
        self._maps: dict[str, dict[str, str]] = {}

    def hset(self, scope: str, key: str, value: str) -> None:
        self._maps.setdefault(scope, {})[key] = value

    def hget(self, scope: str, key: str) -> str | None:
        return self._maps.get(scope, {}).get(key)

    def hgetall(self, scope: str) -> dict[str, str]:
        return dict(self._maps.get(scope, {}))

    def hdel(self, scope: str, key: str) -> None:
        self._maps.get(scope, {}).pop(key, None)


class RedisStore(AgentStore):
    def __init__(self, url: str) -> None:
        import redis  # Imported lazily; unused unless this store is selected.

        self._prefix = _AGENT_SCOPE_PREFIX
        self._redis = redis.Redis.from_url(url, decode_responses=True)

    def hset(self, scope: str, key: str, value: str) -> None:
        self._redis.hset(self._prefix + scope, key, value)

    def hget(self, scope: str, key: str) -> str | None:
        return self._redis.hget(self._prefix + scope, key)

    def hgetall(self, scope: str) -> dict[str, str]:
        return self._redis.hgetall(self._prefix + scope)

    def hdel(self, scope: str, key: str) -> None:
        self._redis.hdel(self._prefix + scope, key)

    def close(self) -> None:
        self._redis.close()


class SqlStore(AgentStore):
    """Hash semantics over any SQLAlchemy dialect (PostgreSQL in production)."""

    def __init__(self, url: str) -> None:
        import sqlalchemy
        from sqlalchemy import (
            Column,
            MetaData,
            String,
            Table,
            Text,
            create_engine,
        )

        self._sqlalchemy = sqlalchemy
        self._engine = create_engine(url)
        self._table = Table(
            "kothagpt_agent_state",
            MetaData(),
            Column("scope", String(64), primary_key=True),
            Column("key", String(255), primary_key=True),
            Column("value", Text, nullable=False),
        )
        self._ready = False

    def _ensure_table(self) -> None:
        if not self._ready:
            self._table.create(self._engine, checkfirst=True)
            self._ready = True

    def hset(self, scope: str, key: str, value: str) -> None:
        table = self._table
        self._ensure_table()
        with self._engine.begin() as conn:
            conn.execute(
                self._sqlalchemy.delete(table).where(
                    (table.c.scope == scope) & (table.c.key == key)
                )
            )
            conn.execute(self._sqlalchemy.insert(table).values(scope=scope, key=key, value=value))

    def hget(self, scope: str, key: str) -> str | None:
        self._ensure_table()
        with self._engine.connect() as conn:
            row = conn.execute(
                self._sqlalchemy.select(self._table.c.value).where(
                    (self._table.c.scope == scope) & (self._table.c.key == key)
                )
            ).first()
        return row[0] if row else None

    def hgetall(self, scope: str) -> dict[str, str]:
        self._ensure_table()
        rows = []
        with self._engine.connect() as conn:
            rows = conn.execute(
                self._sqlalchemy.select(self._table.c.key, self._table.c.value).where(
                    self._table.c.scope == scope
                )
            ).all()
        return {key: value for key, value in rows}

    def hdel(self, scope: str, key: str) -> None:
        self._ensure_table()
        with self._engine.begin() as conn:
            conn.execute(
                self._sqlalchemy.delete(self._table).where(
                    (self._table.c.scope == scope) & (self._table.c.key == key)
                )
            )

    def close(self) -> None:
        self._engine.dispose()


def create_store_from_env(env: dict[str, str] | None = None) -> AgentStore:
    env = os.environ if env is None else env
    url = env.get("KOTHAGPT_AGENT_STORE_URL") or env.get("REDIS_URL")
    if url:
        return RedisStore(url)
    if env.get("DATABASE_URL"):
        return SqlStore(env["DATABASE_URL"])
    return MemoryStore()


class StoredMapping(MutableMapping[Any, Any]):
    """dict-like view over an ``AgentStore`` that serializes pydantic values."""

    def __init__(self, store: AgentStore, scope: str, model: type) -> None:
        self._store = store
        self._scope = scope
        self._model = model

    def __getitem__(self, key: Any) -> Any:
        raw = self._store.hget(self._scope, key)
        if raw is None:
            raise KeyError(key)
        return self._model.model_validate_json(raw)

    def __setitem__(self, key: Any, value: Any) -> None:
        self._store.hset(self._scope, key, value.model_dump_json())

    def __delitem__(self, key: Any) -> None:
        if self._store.hget(self._scope, key) is None:
            raise KeyError(key)
        self._store.hdel(self._scope, key)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._store.hgetall(self._scope))

    def __len__(self) -> int:
        return len(self._store.hgetall(self._scope))

    def values(self) -> list[Any]:
        return [
            self._model.model_validate_json(raw)
            for raw in self._store.hgetall(self._scope).values()
        ]


__all__ = [
    "AgentStore",
    "MemoryStore",
    "RedisStore",
    "SqlStore",
    "StoredMapping",
    "create_store_from_env",
]

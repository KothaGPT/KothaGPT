"""In-memory conversation/message store for the web client.

Conversations and their messages live here so the frontend's
``/v1/conversations/*`` and ``/v1/chat`` endpoints have a place to persist
state across requests within a process. Backed by the same ``AgentStore``
abstraction as agents/runs (MemoryStore by default, Redis/SQL when configured)
so state survives instance restarts on ephemeral hosts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .agent_store import AgentStore, MemoryStore, create_store_from_env


def _cid() -> str:
    return f"conv_{uuid.uuid4().hex[:24]}"


def _mid() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


@dataclass
class StoredMessage:
    id: str
    role: str
    content: str
    model: str | None = None
    created_at: int = field(default_factory=_now)
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    citations: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "createdAt": self.created_at,
            "usage": self.usage,
            "toolCalls": self.tool_calls,
            "citations": self.citations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredMessage:
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            model=data.get("model"),
            created_at=data.get("created_at", _now()),
            usage=data.get("usage"),
            tool_calls=data.get("toolCalls"),
            citations=data.get("citations"),
        )


@dataclass
class StoredConversation:
    id: str
    title: str
    model: str
    created_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "updatedAt": self.updated_at,
            "messageCount": self.message_count,
        }


class ConversationStore:
    """Holds conversations and their messages.

    Uses an ``AgentStore`` for durability (MemoryStore / RedisStore / SqlStore)
    so conversations survive instance restarts. Two scopes:
      - ``conversations`` -> conversation id -> StoredConversation JSON
      - ``messages:<conversation_id>`` -> message id -> StoredMessage JSON
    """

    def __init__(self, store: AgentStore | None = None) -> None:
        self._store = store or MemoryStore()
        self._conv_scope = "conversations"

    # ---- conversations ---------------------------------------------------

    def list(self, limit: int = 50, offset: int = 0) -> list[StoredConversation]:
        raw = self._store.hgetall(self._conv_scope)
        items: list[StoredConversation] = []
        for value in raw.values():
            try:
                data = __import__("json").loads(value)
                items.append(
                    StoredConversation(
                        id=data["id"],
                        title=data["title"],
                        model=data.get("model", "kothagpt"),
                        created_at=data.get("created_at", _now()),
                        updated_at=data.get("updated_at", data.get("created_at", _now())),
                        message_count=data.get("message_count", 0),
                    )
                )
            except Exception:  # noqa: BLE001, S112 — skip corrupt records
                continue
        items.sort(key=lambda c: c.updated_at, reverse=True)
        return items[offset : offset + limit]

    def get(self, conversation_id: str) -> StoredConversation | None:
        raw = self._store.hget(self._conv_scope, conversation_id)
        if raw is None:
            return None
        data = __import__("json").loads(raw)
        return StoredConversation(
            id=data["id"],
            title=data["title"],
            model=data.get("model", "kothagpt"),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", data.get("created_at", _now())),
            message_count=data.get("message_count", 0),
        )

    def create(
        self,
        conversation_id: str,
        title: str,
        model: str = "kothagpt",
    ) -> StoredConversation:
        now = _now()
        conv = StoredConversation(
            id=conversation_id,
            title=title,
            model=model,
            created_at=now,
            updated_at=now,
            message_count=0,
        )
        self._store.hset(
            self._conv_scope, conversation_id, __import__("json").dumps(conv.to_dict())
        )
        return conv

    def delete(self, conversation_id: str) -> bool:
        if self._store.hget(self._conv_scope, conversation_id) is None:
            return False
        # Remove all messages for this conversation.
        msg_scope = f"messages:{conversation_id}"
        for key in list(self._store.hgetall(msg_scope).keys()):
            self._store.hdel(msg_scope, key)
        self._store.hdel(self._conv_scope, conversation_id)
        return True

    def touch(
        self, conversation_id: str, title: str | None = None, model: str | None = None
    ) -> None:
        conv = self.get(conversation_id)
        if conv is None:
            return
        if title is not None:
            conv.title = title
        if model is not None:
            conv.model = model
        conv.updated_at = _now()
        self._store.hset(
            self._conv_scope, conversation_id, __import__("json").dumps(conv.to_dict())
        )

    # ---- messages --------------------------------------------------------

    def _msg_scope(self, conversation_id: str) -> str:
        return f"messages:{conversation_id}"

    def add_message(self, conversation_id: str, message: StoredMessage) -> None:
        self._store.hset(
            self._msg_scope(conversation_id),
            message.id,
            __import__("json").dumps(message.to_dict()),
        )
        conv = self.get(conversation_id)
        if conv is not None:
            conv.message_count += 1
            conv.updated_at = _now()
            self._store.hset(
                self._conv_scope, conversation_id, __import__("json").dumps(conv.to_dict())
            )

    def get_messages(self, conversation_id: str) -> list[StoredMessage]:
        raw = self._store.hgetall(self._msg_scope(conversation_id))
        items: list[StoredMessage] = []
        for value in raw.values():
            try:
                items.append(StoredMessage.from_dict(__import__("json").loads(value)))
            except Exception:  # noqa: BLE001, S112 — skip corrupt records
                continue
        items.sort(key=lambda m: m.created_at)
        return items

    def set_messages(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        scope = self._msg_scope(conversation_id)
        for key in list(self._store.hgetall(scope).keys()):
            self._store.hdel(scope, key)
        for message in messages:
            self._store.hset(scope, message.id, __import__("json").dumps(message.to_dict()))
        conv = self.get(conversation_id)
        if conv is not None:
            conv.message_count = len(messages)
            conv.updated_at = _now()
            self._store.hset(
                self._conv_scope, conversation_id, __import__("json").dumps(conv.to_dict())
            )

    def close(self) -> None:
        self._store.close()


_conversation_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore(create_store_from_env())
    return _conversation_store


__all__ = [
    "ConversationStore",
    "StoredConversation",
    "StoredMessage",
    "get_conversation_store",
]

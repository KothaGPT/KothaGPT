"""WS-2 Conversation management for the web client.

Endpoints:

- ``GET /v1/conversations`` — list conversations (paginated).
- ``GET /v1/conversations/{id}`` — get a conversation.
- ``GET /v1/conversations/{id}/messages`` — list messages in a conversation.
- ``POST /v1/conversations`` — create a conversation.
- ``DELETE /v1/conversations/{id}`` — delete a conversation.

State lives in :class:`services.api.core.conversation_store.ConversationStore`
(MemoryStore by default; Redis/PostgreSQL via ``REDIS_URL``/``DATABASE_URL``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.conversation_store import (
    get_conversation_store,
)

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=200)
    model: str = Field(default="kothagpt")


class ConversationOut(BaseModel):
    id: str
    title: str
    model: str
    updatedAt: int
    messageCount: int


class ConversationList(BaseModel):
    object: str = "list"
    data: list[ConversationOut]
    total: int


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    model: str | None = None
    createdAt: int | None = None
    usage: dict[str, Any] | None = None
    toolCalls: list[dict[str, Any]] | None = None
    citations: list[str] | None = None


@router.get("", response_model=ConversationList)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ConversationList:
    store = get_conversation_store()
    items = store.list(limit=limit, offset=offset)
    total = len(store.list(limit=10_000))
    return ConversationList(
        data=[
            ConversationOut(
                id=c.id,
                title=c.title,
                model=c.model,
                updatedAt=c.updated_at,
                messageCount=c.message_count,
            )
            for c in items
        ],
        total=total,
    )


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str) -> ConversationOut:
    store = get_conversation_store()
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        updatedAt=conv.updated_at,
        messageCount=conv.message_count,
    )


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: str) -> list[MessageOut]:
    store = get_conversation_store()
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            model=m.model,
            createdAt=m.created_at,
            usage=m.usage,
            toolCalls=m.tool_calls,
            citations=m.citations,
        )
        for m in store.get_messages(conversation_id)
    ]


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(payload: ConversationCreate) -> ConversationOut:
    store = get_conversation_store()
    conversation_id = _cid()
    conv = store.create(conversation_id, payload.title, payload.model)
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        updatedAt=conv.updated_at,
        messageCount=conv.message_count,
    )


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    store = get_conversation_store()
    if not store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")


def _cid() -> str:
    import uuid

    return f"conv_{uuid.uuid4().hex[:24]}"


__all__ = ["router"]

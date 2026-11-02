from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.backend import BackendUnavailableError
from ...core.conversation_store import (
    StoredMessage,
    get_conversation_store,
)
from ...core.rag_runtime import rag_runtime
from ...router import route_request
from ..backend import get_active_backend
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LegacyChatResponse,
    Message,
)

router = APIRouter(prefix="/v1", tags=["chat"])

# WS-5: explicit tier pin via X-Kotha-Tier header. "auto" (or absent) lets the
# ModelRouter pick the cheapest tier that meets the quality floor.
_VALID_TIERS = {"auto", "nano", "small", "medium", "large"}
_VALID_RAG = {"true", "false", "1", "0", "yes", "no"}


def _apply_rag_grounding(
    request: ChatCompletionRequest,
    x_kotha_rag: str | None,
    top_k: int,
) -> tuple[ChatCompletionRequest, list[dict] | None]:
    """When RAG grounding is requested, retrieve context for the user prompt and
    prepend a grounded system message. Returns the (possibly modified) request
    and the citation list to surface in a response header.
    """
    enabled = (x_kotha_rag or "").strip().lower()
    if enabled not in _VALID_RAG or enabled in {"false", "0", "no"}:
        return request, None
    prompt = "\n".join(m.content for m in request.messages if m.role == "user")
    if not prompt.strip():
        return request, None
    context_text, citations = rag_runtime.build_context(prompt, top_k=top_k)
    if not context_text.strip():
        return request, None
    system_msg = (
        "You are a helpful assistant. Answer using ONLY the following retrieved "
        "context when it is relevant. Cite sources with the bracketed markers "
        "(e.g. [1], [2]) when you use them. If the context does not contain the "
        "answer, say so instead of making things up.\n\n"
        f"Context:\n{context_text}"
    )
    grounded_messages = [
        Message(role="system", content=system_msg),
        *request.messages,
    ]
    grounded_request = ChatCompletionRequest(
        model=request.model,
        messages=grounded_messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        user=request.user,
    )
    return grounded_request, citations


def _chat_completion_request(
    messages: list[dict[str, Any]],
    model: str = "kothagpt",
    temperature: float | None = 0.7,
    top_p: float | None = 1.0,
    max_tokens: int | None = None,
    stream: bool = False,
) -> ChatCompletionRequest:
    """Build a :class:`ChatCompletionRequest` from raw message dicts."""
    chat_messages = [
        Message(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            name=m.get("name"),
            tool_calls=m.get("toolCalls"),
            tool_call_id=m.get("toolCallId"),
        )
        for m in messages
    ]
    return ChatCompletionRequest(
        model=model,
        messages=chat_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=stream,
    )


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, alias="conversationId")
    model: str = "kothagpt"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, alias="maxTokens", ge=1)
    stream: bool = False

    model_config = {"populate_by_name": True}


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    model: str | None = None
    created_at: int | None = Field(default=None, alias="createdAt")


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    x_kotha_tier: str | None = Header(default=None, alias="X-Kotha-Tier"),
    x_kotha_task: str | None = Header(default=None, alias="X-Kotha-Task"),
    x_kotha_rag: str | None = Header(default=None, alias="X-Kotha-RAG"),
    x_kotha_rag_top_k: str | None = Header(default=None, alias="X-Kotha-RAG-Top-K"),
) -> StreamingResponse | ChatCompletionResponse:
    # WS-5 routing: resolve the tier before dispatching. The resolved tier is
    # attached to the response headers so backends can log/observe it.
    if x_kotha_tier is not None:
        normalized = x_kotha_tier.strip().lower()
        if normalized not in _VALID_TIERS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown tier {normalized!r}; expected one of {sorted(_VALID_TIERS)}",
            )
        x_kotha_tier = normalized

    # WS-9 RAG grounding: optionally retrieve context and prefix it as a system
    # message so answers are grounded in the indexed knowledge base.
    try:
        top_k = int(x_kotha_rag_top_k or 5)
    except ValueError:
        top_k = 5
    request, citations = _apply_rag_grounding(request, x_kotha_rag, top_k)

    prompt = "\n".join(m.content for m in request.messages if m.role == "user")
    routing = route_request(
        prompt,
        x_kotha_tier=x_kotha_tier,
        x_kotha_task=x_kotha_task,
        expected_tokens=request.max_tokens or 64,
    )
    response.headers["X-Kotha-Routing"] = json.dumps(routing, ensure_ascii=False)
    if citations:
        response.headers["X-Kotha-RAG-Citations"] = json.dumps(citations, ensure_ascii=False)

    if request.stream:
        backend = get_active_backend("chat", "streaming")
        return StreamingResponse(
            _sse_stream(backend.chat_stream(request)),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Kotha-Routing": json.dumps(routing, ensure_ascii=False),
                **(
                    {"X-Kotha-RAG-Citations": json.dumps(citations, ensure_ascii=False)}
                    if citations
                    else {}
                ),
            },
        )
    backend = get_active_backend("chat")
    try:
        return backend.chat(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatCompletionRequest,
    x_kotha_rag: str | None = Header(default=None, alias="X-Kotha-RAG"),
    x_kotha_rag_top_k: str | None = Header(default=None, alias="X-Kotha-RAG-Top-K"),
) -> StreamingResponse:
    """Convenience endpoint: always streams, regardless of the stream flag."""
    try:
        top_k = int(x_kotha_rag_top_k or 5)
    except ValueError:
        top_k = 5
    request, citations = _apply_rag_grounding(request, x_kotha_rag, top_k)

    backend = get_active_backend("chat", "streaming")
    headers: dict[str, str] = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if citations:
        headers["X-Kotha-RAG-Citations"] = json.dumps(citations, ensure_ascii=False)
    return StreamingResponse(
        _sse_stream(backend.chat_stream(request)),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/chat", response_model=None)
async def chat(
    request: Request,
    response: Response,
) -> StreamingResponse | ChatMessageOut | LegacyChatResponse:
    """Unified chat endpoint.

    Accepts either the legacy single-message body::

        {"message": "...", "model": "kothagpt"}

    or the web-client ``ChatRequest`` body::

        {"conversationId": "...", "model": "kothagpt",
         "messages": [{role, content}], "stream": true, ...}

    When ``stream`` is true, emits ``ChatStreamEvent``-shaped SSE frames
    (``start``/``delta``/``done``). Conversations are persisted via the
    :class:`services.api.core.conversation_store.ConversationStore`.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Legacy single-message path.
    if "message" in body and "messages" not in body:
        backend = get_active_backend("chat")
        chat_req = ChatCompletionRequest(
            model=body.get("model", "kothagpt"),
            messages=[{"role": "user", "content": str(body.get("message", ""))}],
        )
        try:
            resp = backend.chat(chat_req)
        except BackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return LegacyChatResponse(
            model=chat_req.model,
            message=chat_req.messages[0].content,
            output=resp.choices[0].message.content,
        )

    # Web-client path.
    try:
        chat_request = ChatRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid chat request: {exc}") from exc

    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    conversation_id = chat_request.conversation_id
    store = get_conversation_store()

    # Ensure the conversation exists (create it from the first user message if needed).
    if conversation_id is None:
        first_user = next(
            (m.get("content", "") for m in chat_request.messages if m.get("role") == "user"),
            "",
        )
        conversation_id = store.create(
            _cid(),
            title=(first_user[:60] or "New chat"),
            model=chat_request.model,
        ).id
    elif store.get(conversation_id) is None:
        store.create(conversation_id, title="New chat", model=chat_request.model)

    # Persist the incoming messages (client is authoritative for the transcript).
    stored = [
        StoredMessage(
            id=m.get("id") or _mid(),
            role=m.get("role", "user"),
            content=m.get("content", ""),
            model=m.get("model") or chat_request.model,
        )
        for m in chat_request.messages
    ]
    store.set_messages(conversation_id, stored)

    completion_request = _chat_completion_request(
        messages=chat_request.messages,
        model=chat_request.model,
        temperature=chat_request.temperature,
        max_tokens=chat_request.max_tokens,
        stream=True,
    )
    if chat_request.system_prompt:
        completion_request.messages.insert(
            0,
            Message(role="system", content=chat_request.system_prompt),
        )

    backend = get_active_backend("chat", "streaming")

    assistant_id = _mid()
    assistant_content_parts: list[str] = []

    def generate():
        # start event
        yield f"data: {json.dumps({'type': 'start', 'conversationId': conversation_id}, ensure_ascii=False)}\n\n"
        try:
            for chunk in backend.chat_stream(completion_request):
                choices = chunk.choices or []
                if choices:
                    delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                    content = delta.get("content", "") or ""
                    if content:
                        assistant_content_parts.append(content)
                        yield f"data: {json.dumps({'type': 'delta', 'content': content}, ensure_ascii=False)}\n\n"
        except BackendUnavailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        assistant_content = "".join(assistant_content_parts)
        # Persist the assistant message.
        assistant_msg = StoredMessage(
            id=assistant_id,
            role="assistant",
            content=assistant_content,
            model=completion_request.model,
        )
        current = store.get_messages(conversation_id)
        current.append(assistant_msg)
        store.set_messages(conversation_id, current)
        store.touch(conversation_id)
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/chat/{conversation_id}/message/{message_id}", response_model=ChatMessageOut)
def get_message(conversation_id: str, message_id: str) -> ChatMessageOut:
    store = get_conversation_store()
    if store.get(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    for message in store.get_messages(conversation_id):
        if message.id == message_id:
            return ChatMessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                model=message.model,
                created_at=message.created_at,
            )
    raise HTTPException(status_code=404, detail="Message not found")


def _sse_stream(chunks) -> StreamingResponse:  # type: ignore[no-untyped-def]
    def generate():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return generate()


def _cid() -> str:
    return f"conv_{uuid.uuid4().hex[:24]}"


def _mid() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


__all__ = ["router"]

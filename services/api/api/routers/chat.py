from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from ...core.backend import BackendUnavailableError
from ...core.rag_runtime import rag_runtime
from ...router import route_request
from ..backend import get_active_backend
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LegacyChatRequest,
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


@router.post("/chat", response_model=LegacyChatResponse)
async def chat(request: LegacyChatRequest) -> LegacyChatResponse:
    """Legacy single-message chat, kept for backward compatibility."""
    backend = get_active_backend("chat")
    chat_req = ChatCompletionRequest(
        model=request.model,
        messages=[{"role": "user", "content": request.message}],
    )
    try:
        response = backend.chat(chat_req)
    except BackendUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LegacyChatResponse(
        model=request.model,
        message=request.message,
        output=response.choices[0].message.content,
    )


def _sse_stream(chunks) -> StreamingResponse:  # type: ignore[no-untyped-def]
    def generate():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return generate()


__all__ = ["router"]

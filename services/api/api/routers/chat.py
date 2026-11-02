from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from ...core.backend import BackendUnavailableError
from ...router import route_request
from ..backend import get_active_backend
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LegacyChatRequest,
    LegacyChatResponse,
)

router = APIRouter(prefix="/v1", tags=["chat"])

# WS-5: explicit tier pin via X-Kotha-Tier header. "auto" (or absent) lets the
# ModelRouter pick the cheapest tier that meets the quality floor.
_VALID_TIERS = {"auto", "nano", "small", "medium", "large"}


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    x_kotha_tier: str | None = Header(default=None, alias="X-Kotha-Tier"),
    x_kotha_task: str | None = Header(default=None, alias="X-Kotha-Task"),
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
    prompt = "\n".join(m.content for m in request.messages if m.role == "user")
    routing = route_request(
        prompt,
        x_kotha_tier=x_kotha_tier,
        x_kotha_task=x_kotha_task,
        expected_tokens=request.max_tokens or 64,
    )
    response.headers["X-Kotha-Routing"] = json.dumps(routing, ensure_ascii=False)

    if request.stream:
        backend = get_active_backend("chat", "streaming")
        return StreamingResponse(
            _sse_stream(backend.chat_stream(request)),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Kotha-Routing": json.dumps(routing, ensure_ascii=False),
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
async def chat_stream(request: ChatCompletionRequest) -> StreamingResponse:
    """Convenience endpoint: always streams, regardless of the stream flag."""
    backend = get_active_backend("chat", "streaming")
    return StreamingResponse(
        _sse_stream(backend.chat_stream(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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

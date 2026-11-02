from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from ...core import backend_factory
from ..auth import require_api_token_websocket
from ..backend import get_active_backend
from ..schemas import (
    AgentSpec,
    ChatCompletionRequest,
    EmbeddingRequest,
    RerankRequest,
    ToolInvokeRequest,
    WsEnvelope,
)

router = APIRouter(
    tags=["websocket"],
    dependencies=[Depends(require_api_token_websocket)],
)

_HANDLERS = {
    "ping": lambda p: {"pong": True},
    "models.list": lambda p: {
        "data": [m.model_dump() for m in backend_factory.create().list_models()]
    },
}


@router.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                envelope = WsEnvelope.model_validate(json.loads(raw))
            except Exception:  # noqa: BLE001
                await _send(
                    websocket, envelope_id=None, response={"error": "invalid message"}, kind="error"
                )
                continue

            try:
                response = await _dispatch(envelope)
            except HTTPException as exc:
                response = {"error": exc.detail, "status_code": exc.status_code}
            await _send(websocket, envelope.id, response, kind=envelope.type)
    except WebSocketDisconnect:
        return


async def _dispatch(envelope: WsEnvelope) -> dict:
    kind, payload = envelope.type, envelope.payload
    if kind in _HANDLERS:
        return _HANDLERS[kind](payload)
    if kind == "tools.list":
        backend = get_active_backend("tools")
        return {"data": [t.model_dump() for t in backend.list_tools()]}
    if kind == "chat":
        request = ChatCompletionRequest.model_validate(payload)
        capabilities = ("chat", "streaming") if request.stream else ("chat",)
        backend = get_active_backend(*capabilities)
        if request.stream:
            chunks = []
            for chunk in backend.chat_stream(request):
                chunks.append(chunk.model_dump())
            return {"chunks": chunks}
        return backend.chat(request).model_dump()
    if kind == "embed":
        backend = get_active_backend("embeddings")
        request = EmbeddingRequest.model_validate(payload)
        inputs = request.input if isinstance(request.input, list) else [request.input]
        return backend.embed(request.model, inputs).model_dump()
    if kind == "rerank":
        backend = get_active_backend("rerank")
        request = RerankRequest.model_validate(payload)
        return backend.rerank(
            request.model, request.query, request.documents, request.top_n
        ).model_dump()
    if kind == "tools.invoke":
        backend = get_active_backend("tools")
        request = ToolInvokeRequest.model_validate(payload)
        return backend.invoke_tool(request.name, request.arguments).model_dump()
    if kind == "agents.list":
        backend = get_active_backend("agents")
        return {"data": [a.model_dump() for a in backend.list_agents()]}
    if kind == "agents.create":
        backend = get_active_backend("agents")
        spec = AgentSpec.model_validate(payload)
        return backend.create_agent(spec).model_dump()
    if kind == "agents.run":
        backend = get_active_backend("agents")
        agent_id = payload.get("agent_id")
        message = payload.get("message", "")
        if agent_id is None:
            return {"error": "agent_id required"}
        if payload.get("stream"):
            chunks = [e async for e in backend.run_agent_stream(agent_id, message)]
            return {"events": chunks}
        return backend.run_agent(agent_id, message).model_dump()
    return {"error": f"unsupported type: {kind}"}


async def _send(websocket: WebSocket, envelope_id: str | None, response: dict, kind: str) -> None:
    await websocket.send_text(
        json.dumps(
            {
                "id": envelope_id,
                "type": kind,
                "payload": response,
            },
            ensure_ascii=False,
        )
    )

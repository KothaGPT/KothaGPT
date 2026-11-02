from __future__ import annotations

import json
from typing import Any, Self

from .errors import APIError
from .types import Agent, AgentRun, ChatCompletion, EmbeddingResponse, RerankResponse

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError:  # pragma: no cover
    websockets = None
    ClientConnection = Any  # type: ignore[misc,assignment]


class WebSocketClient:
    """Low-level JSON-over-WebSocket client for the Kotha GPT /v1/ws endpoint."""

    def __init__(self, base_url: str = "ws://localhost:8000", api_key: str | None = None) -> None:
        if websockets is None:  # pragma: no cover
            raise APIError(
                0, "The 'websockets' package is required. pip install kothagpt[websockets]"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._conn: ClientConnection | None = None

    async def connect(self) -> WebSocketClient:
        import websockets

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._conn = await websockets.connect(f"{self.base_url}/v1/ws", additional_headers=headers)
        return self

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _request(
        self, type: str, payload: dict[str, Any], id: str | None = None
    ) -> dict[str, Any]:
        if self._conn is None:
            raise APIError(0, "Not connected; call connect() first")
        await self._conn.send(json.dumps({"id": id, "type": type, "payload": payload}))
        raw = await self._conn.recv()
        message = json.loads(raw) if isinstance(raw, str) else raw
        return message

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
        reply = await self._request("chat", {"messages": messages, **kwargs})
        return ChatCompletion.model_validate(reply["payload"])

    async def embed(
        self, input: str | list[str], model: str = "kothagpt-embed"
    ) -> EmbeddingResponse:
        reply = await self._request("embed", {"input": input, "model": model})
        return EmbeddingResponse.model_validate(reply["payload"])

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> RerankResponse:
        reply = await self._request(
            "rerank", {"query": query, "documents": documents, "top_n": top_n}
        )
        return RerankResponse.model_validate(reply["payload"])

    async def tools_list(self) -> list[dict[str, Any]]:
        reply = await self._request("tools.list", {})
        return reply["payload"]["data"]

    async def tools_invoke(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        reply = await self._request("tools.invoke", {"name": name, "arguments": arguments or {}})
        return reply["payload"]["result"]

    async def agents_create(self, spec: dict[str, Any]) -> Agent:
        reply = await self._request("agents.create", spec)
        return Agent.model_validate(reply["payload"])

    async def agents_list(self) -> list[Agent]:
        reply = await self._request("agents.list", {})
        return [Agent.model_validate(a) for a in reply["payload"]["data"]]

    async def agents_run(self, agent_id: str, message: str) -> AgentRun:
        reply = await self._request("agents.run", {"agent_id": agent_id, "message": message})
        return AgentRun.model_validate(reply["payload"])

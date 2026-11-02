from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Self

from . import types
from ._http import AsyncMixin, SyncMixin, _sse_lines, build_params, parse_sse, raise_for_status


class BaseKothaGPT:
    """Shared API-surface definitions for the sync and async clients.

    Each method is implemented by the sync/async mixins; this class only
    documents the supported operations and builds request payloads.
    """

    base_url: str
    api_key: str | None

    @staticmethod
    def _base_headers() -> dict[str, str]:
        return {"User-Agent": "kothagpt-python-sdk/0.1.0"}

    def _chat_payload(
        self,
        messages: list[types.ChatMessage | dict[str, Any]],
        *,
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[types.Tool] | None,
        stream: bool,
    ) -> dict[str, Any]:
        return build_params(
            model=model,
            messages=[m.model_dump() if isinstance(m, types.ChatMessage) else m for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=stream,
        )


class KothaGPT(SyncMixin, BaseKothaGPT):
    """Sync client for the Kotha GPT API.

    ```python
    client = KothaGPT(api_key="...")
    completion = client.chat.completions.create(messages=[{"role": "user", "content": "হ্যালো"}])
    print(completion.text)
    ```
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat = ChatCompletions(self)
        self.embeddings = Embeddings(self)
        self.rerank = Rerank(self)
        self.models = Models(self)
        self.tools = Tools(self)
        self.agents = Agents(self)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncKothaGPT(AsyncMixin, BaseKothaGPT):
    """Async client for the Kotha GPT API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat = AsyncChatCompletions(self)
        self.embeddings = AsyncEmbeddings(self)
        self.rerank = AsyncRerank(self)
        self.models = AsyncModels(self)
        self.tools = AsyncTools(self)
        self.agents = AsyncAgents(self)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class ChatCompletions:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def create(
        self,
        messages: list[types.ChatMessage | dict[str, Any]],
        *,
        model: str = "kothagpt",
        temperature: float | None = 0.7,
        top_p: float | None = 1.0,
        max_tokens: int | None = None,
        tools: list[types.Tool] | None = None,
        stream: bool = False,
    ) -> types.ChatCompletion:
        payload = self._client._chat_payload(  # type: ignore[attr-defined]
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            stream=stream,
        )
        response = self._client._post("/v1/chat/completions", payload)  # type: ignore[attr-defined]
        return types.ChatCompletion.model_validate(response.json())

    def stream(
        self,
        messages: list[types.ChatMessage | dict[str, Any]],
        *,
        model: str = "kothagpt",
        temperature: float | None = 0.7,
        top_p: float | None = 1.0,
        max_tokens: int | None = None,
        tools: list[types.Tool] | None = None,
    ) -> Iterator[types.ChatChunk]:
        payload = self._client._chat_payload(  # type: ignore[attr-defined]
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
        )
        with self._client._stream("/v1/chat/completions", payload) as response:  # type: ignore[attr-defined]
            raise_for_status(response)
            yield from parse_sse(response, types.ChatChunk)


class Embeddings:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def create(
        self, input: str | list[str], *, model: str = "kothagpt-embed"
    ) -> types.EmbeddingResponse:
        response = self._client._post("/v1/embeddings", {"model": model, "input": input})  # type: ignore[attr-defined]
        return types.EmbeddingResponse.model_validate(response.json())


class Rerank:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def create(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "kothagpt-rerank",
        top_n: int | None = None,
    ) -> types.RerankResponse:
        response = self._client._post(  # type: ignore[attr-defined]
            "/v1/rerank", build_params(model=model, query=query, documents=documents, top_n=top_n)
        )
        return types.RerankResponse.model_validate(response.json())


class Models:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def list(self) -> list[types.Model]:
        response = self._client._get("/v1/models")  # type: ignore[attr-defined]
        return [types.Model.model_validate(m) for m in response.json()["data"]]


class Tools:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def list(self) -> list[types.Tool]:
        response = self._client._get("/v1/tools")  # type: ignore[attr-defined]
        return [types.Tool.model_validate(t) for t in response.json()["data"]]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        response = self._client._post(
            f"/v1/tools/{name}/invoke", {"name": name, "arguments": arguments or {}}
        )  # type: ignore[attr-defined]
        return response.json()["result"]


class Agents:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    def create(self, spec: types.AgentSpec | dict[str, Any]) -> types.Agent:
        payload = spec.model_dump() if isinstance(spec, types.AgentSpec) else spec
        response = self._client._post("/v1/agents", payload)  # type: ignore[attr-defined]
        return types.Agent.model_validate(response.json())

    def get(self, agent_id: str) -> types.Agent:
        response = self._client._get(f"/v1/agents/{agent_id}")  # type: ignore[attr-defined]
        return types.Agent.model_validate(response.json())

    def list(self) -> list[types.Agent]:
        response = self._client._get("/v1/agents")  # type: ignore[attr-defined]
        return [types.Agent.model_validate(a) for a in response.json()["data"]]

    def delete(self, agent_id: str) -> None:
        self._client._delete(f"/v1/agents/{agent_id}")  # type: ignore[attr-defined]

    def run(self, agent_id: str, message: str) -> types.AgentRun:
        response = self._client._post(f"/v1/agents/{agent_id}/runs", {"message": message})  # type: ignore[attr-defined]
        return types.AgentRun.model_validate(response.json())

    def stream(self, agent_id: str, message: str) -> Iterator[dict[str, Any]]:
        with self._client._stream(
            f"/v1/agents/{agent_id}/runs/stream", {"message": message}
        ) as response:  # type: ignore[attr-defined]
            raise_for_status(response)
            for data in _sse_lines(response):
                yield json.loads(data)

    def get_run(self, agent_id: str, run_id: str) -> types.AgentRun:
        response = self._client._get(f"/v1/agents/{agent_id}/runs/{run_id}")  # type: ignore[attr-defined]
        return types.AgentRun.model_validate(response.json())


class AsyncChatCompletions:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def create(
        self,
        messages: list[types.ChatMessage | dict[str, Any]],
        *,
        model: str = "kothagpt",
        temperature: float | None = 0.7,
        top_p: float | None = 1.0,
        max_tokens: int | None = None,
        tools: list[types.Tool] | None = None,
        stream: bool = False,
    ) -> types.ChatCompletion:
        payload = self._client._chat_payload(  # type: ignore[attr-defined]
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            stream=stream,
        )
        response = await self._client._post("/v1/chat/completions", payload)  # type: ignore[attr-defined]
        return types.ChatCompletion.model_validate(response.json())

    async def stream(
        self,
        messages: list[types.ChatMessage | dict[str, Any]],
        *,
        model: str = "kothagpt",
        temperature: float | None = 0.7,
        top_p: float | None = 1.0,
        max_tokens: int | None = None,
        tools: list[types.Tool] | None = None,
    ):
        payload = self._client._chat_payload(  # type: ignore[attr-defined]
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
        )
        response = await self._client._stream("/v1/chat/completions", payload)  # type: ignore[attr-defined]
        async with response as resp:
            raise_for_status(resp)
            async for data in _aiter_sse(resp):
                yield types.ChatChunk.model_validate(data)


class AsyncEmbeddings:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def create(
        self, input: str | list[str], *, model: str = "kothagpt-embed"
    ) -> types.EmbeddingResponse:
        response = await self._client._post("/v1/embeddings", {"model": model, "input": input})  # type: ignore[attr-defined]
        return types.EmbeddingResponse.model_validate(response.json())


class AsyncRerank:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def create(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "kothagpt-rerank",
        top_n: int | None = None,
    ) -> types.RerankResponse:
        response = await self._client._post(  # type: ignore[attr-defined]
            "/v1/rerank", build_params(model=model, query=query, documents=documents, top_n=top_n)
        )
        return types.RerankResponse.model_validate(response.json())


class AsyncModels:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def list(self) -> list[types.Model]:
        response = await self._client._get("/v1/models")  # type: ignore[attr-defined]
        return [types.Model.model_validate(m) for m in response.json()["data"]]


class AsyncTools:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def list(self) -> list[types.Tool]:
        response = await self._client._get("/v1/tools")  # type: ignore[attr-defined]
        return [types.Tool.model_validate(t) for t in response.json()["data"]]

    async def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        response = await self._client._post(
            f"/v1/tools/{name}/invoke", {"name": name, "arguments": arguments or {}}
        )  # type: ignore[attr-defined]
        return response.json()["result"]


class AsyncAgents:
    def __init__(self, client: BaseKothaGPT) -> None:
        self._client = client

    async def create(self, spec: types.AgentSpec | dict[str, Any]) -> types.Agent:
        payload = spec.model_dump() if isinstance(spec, types.AgentSpec) else spec
        response = await self._client._post("/v1/agents", payload)  # type: ignore[attr-defined]
        return types.Agent.model_validate(response.json())

    async def get(self, agent_id: str) -> types.Agent:
        response = await self._client._get(f"/v1/agents/{agent_id}")  # type: ignore[attr-defined]
        return types.Agent.model_validate(response.json())

    async def list(self) -> list[types.Agent]:
        response = await self._client._get("/v1/agents")  # type: ignore[attr-defined]
        return [types.Agent.model_validate(a) for a in response.json()["data"]]

    async def delete(self, agent_id: str) -> None:
        await self._client._delete(f"/v1/agents/{agent_id}")  # type: ignore[attr-defined]

    async def run(self, agent_id: str, message: str) -> types.AgentRun:
        response = await self._client._post(f"/v1/agents/{agent_id}/runs", {"message": message})  # type: ignore[attr-defined]
        return types.AgentRun.model_validate(response.json())

    async def stream(self, agent_id: str, message: str):
        response = await self._client._stream(
            f"/v1/agents/{agent_id}/runs/stream", {"message": message}
        )  # type: ignore[attr-defined]
        async with response as resp:
            raise_for_status(resp)
            async for data in _aiter_sse(resp):
                yield data

    async def get_run(self, agent_id: str, run_id: str) -> types.AgentRun:
        response = await self._client._get(f"/v1/agents/{agent_id}/runs/{run_id}")  # type: ignore[attr-defined]
        return types.AgentRun.model_validate(response.json())


def _aiter_sse(response):
    import json

    async def generate():
        async for raw in response.aiter_lines():
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)

    return generate()

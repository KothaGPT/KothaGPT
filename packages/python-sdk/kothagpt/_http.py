from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from .errors import raise_for_status


def _sse_lines(response: httpx.Response) -> Iterator[str]:
    """Yield SSE `data:` payloads from a streaming response."""
    for raw in response.iter_lines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        yield data


def parse_sse(response: httpx.Response, model: type) -> Iterator[Any]:
    for data in _sse_lines(response):
        yield model.model_validate(json.loads(data))


def build_params(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


class SyncMixin:
    _client: httpx.Client

    def _post(self, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        response = self._client.post(path, json=json)
        raise_for_status(response)
        return response

    def _get(self, path: str) -> httpx.Response:
        response = self._client.get(path)
        raise_for_status(response)
        return response

    def _delete(self, path: str) -> httpx.Response:
        response = self._client.delete(path)
        raise_for_status(response)
        return response

    def _stream(self, path: str, json: dict[str, Any]):
        return self._client.stream("POST", path, json=json)


class AsyncMixin:
    _client: httpx.AsyncClient

    async def _post(self, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        response = await self._client.post(path, json=json)
        raise_for_status(response)
        return response

    async def _get(self, path: str) -> httpx.Response:
        response = await self._client.get(path)
        raise_for_status(response)
        return response

    async def _delete(self, path: str) -> httpx.Response:
        response = await self._client.delete(path)
        raise_for_status(response)
        return response

    async def _stream(self, path: str, json: dict[str, Any]):
        return self._client.stream("POST", path, json=json)

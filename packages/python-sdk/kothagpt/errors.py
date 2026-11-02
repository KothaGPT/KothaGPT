from __future__ import annotations

from typing import Any

import httpx


class KothaGPTError(Exception):
    """Base exception for all SDK errors."""


class APIError(KothaGPTError):
    """The API returned a non-2xx response."""

    def __init__(self, status_code: int, message: str, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body


class APIStatusError(APIError):
    """The API returned an unexpected status code."""


class APIConnectionError(KothaGPTError):
    """The request could not reach the API."""


class AuthenticationError(APIStatusError):
    """The API rejected the API key."""


class NotFoundError(APIStatusError):
    """The requested resource was not found."""


def raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    body = None
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = None
    message = body.get("detail") if isinstance(body, dict) else None
    message = message or f"Request failed with status {response.status_code}"
    if response.status_code == 401:
        raise AuthenticationError(response.status_code, message, body)
    if response.status_code == 404:
        raise NotFoundError(response.status_code, message, body)
    raise APIStatusError(response.status_code, message, body)

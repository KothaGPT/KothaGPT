from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Query, WebSocket, WebSocketException, status

from services.api.core.config import RuntimeConfig


async def require_api_token(authorization: str | None = Header(default=None)) -> str:
    """Require a bearer token when the API is not explicitly in local mode."""
    cfg = RuntimeConfig.from_env()
    if cfg.api_token is None:
        if cfg.local_mode:
            return "local"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(token, cfg.api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )
    return token


ApiToken = Annotated[str, require_api_token]


async def require_api_token_websocket(
    websocket: WebSocket,
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Enforce the same bearer-token rules on WebSocket handshakes.

    Browsers cannot set custom headers during the handshake, so a ``token``
    query parameter is accepted as an alternative to the Authorization header.
    """
    cfg = RuntimeConfig.from_env()
    supplied = None
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied and token:
        supplied = token.strip()

    if cfg.api_token is None:
        if cfg.local_mode:
            return
    elif supplied and secrets.compare_digest(supplied, cfg.api_token):
        return
    raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


__all__ = ["ApiToken", "require_api_token", "require_api_token_websocket"]

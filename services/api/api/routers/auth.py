"""WS-1 Authentication surface for the web client.

Lightweight, dependency-free auth so the dashboard can authenticate against
the API without a database. The API token (``KOTHAGPT_API_TOKEN``) is the
single source of truth; a local "dev" session is issued when the client
presents a valid bearer token.

Endpoints:

- ``GET /v1/me`` — current user (requires a valid bearer token unless local
  mode is enabled).
- ``POST /v1/auth/login`` — issue a dev session for a valid bearer token.
- ``PUT /v1/auth/logout`` — clear the session (client-side token clearing).
- ``POST /v1/auth/register`` / ``POST /v1/auth/forgot-password`` — accepted
  but return unimplemented placeholders (no user DB in Phase 3).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status

from ...core.config import RuntimeConfig

router = APIRouter(prefix="/v1", tags=["auth"])


class UserOut(dict):
    pass


def _dev_user() -> dict[str, Any]:
    return {
        "id": "dev-user-1",
        "name": "Developer",
        "email": "dev@kothagpt.local",
        "avatarUrl": None,
    }


def _session(access_token: str) -> dict[str, Any]:
    import time

    return {
        "user": _dev_user(),
        "accessToken": access_token,
        "expiresAt": str(int(time.time()) + 60 * 60 * 24 * 30),
    }


def _require_token(authorization: str | None) -> str:
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
    if not token or token != cfg.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )
    return token


@router.get("/me")
def me(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_token(authorization)
    return _dev_user()


@router.post("/auth/login")
def login(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _require_token(authorization)
    return _session(token)


@router.put("/auth/logout")
def logout() -> dict[str, str]:
    return {"status": "logged out"}


@router.post("/auth/register")
def register() -> dict[str, Any]:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User registration is not available without a user database.",
    )


@router.post("/auth/forgot-password")
def forgot_password() -> dict[str, str]:
    return {"status": "password-reset-not-available"}


__all__ = ["router"]

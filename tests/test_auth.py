"""WS-12: API authentication via ``require_api_token`` dependency."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.api.api.auth import require_api_token


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/auth-test")
    async def auth_test(_token: str = Depends(require_api_token)):
        return {"ok": True}

    return app


def test_auth_passes_with_correct_token(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", "correct-token")
    monkeypatch.delenv("KOTHAGPT_LOCAL_MODE", raising=False)
    client = TestClient(_make_app())
    r = client.get("/auth-test", headers={"Authorization": "Bearer correct-token"})
    assert r.status_code == 200, r.text


def test_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", "correct-token")
    monkeypatch.delenv("KOTHAGPT_LOCAL_MODE", raising=False)
    client = TestClient(_make_app())
    r = client.get("/auth-test", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_auth_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", "correct-token")
    monkeypatch.delenv("KOTHAGPT_LOCAL_MODE", raising=False)
    client = TestClient(_make_app())
    r = client.get("/auth-test")
    assert r.status_code == 401


def test_auth_allows_local_mode_without_token(monkeypatch):
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "true")
    client = TestClient(_make_app())
    r = client.get("/auth-test")
    assert r.status_code == 200, r.text


def test_auth_503_when_not_configured_and_not_local(monkeypatch):
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.delenv("KOTHAGPT_LOCAL_MODE", raising=False)
    client = TestClient(_make_app())
    r = client.get("/auth-test")
    assert r.status_code == 503

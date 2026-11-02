"""WS-13: rate-limit middleware returns 429 with Retry-After when a key exceeds its window."""

from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.app import app
from services.api.rate_limit import limiter


def _make_client(monkeypatch, token: str = "test-token") -> TestClient:
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", token)
    monkeypatch.delenv("KOTHAGPT_LOCAL_MODE", raising=False)
    return TestClient(app)


def test_rate_limit_middleware_returns_429_with_retry_after(monkeypatch):
    limiter.max_requests = 2
    limiter.window = 60
    limiter.reset()
    client = _make_client(monkeypatch)
    headers = {"Authorization": "Bearer test-token"}

    for _ in range(2):
        r = client.get("/health", headers=headers)
        assert r.status_code == 200, r.text

    r = client.get("/health", headers=headers)
    assert r.status_code == 429, r.text
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) > 0


def test_rate_limit_middleware_allows_after_reset(monkeypatch):
    limiter.max_requests = 1
    limiter.window = 60
    limiter.reset()
    client = _make_client(monkeypatch)
    headers = {"Authorization": "Bearer test-token"}

    r = client.get("/health", headers=headers)
    assert r.status_code == 200, r.text

    r = client.get("/health", headers=headers)
    assert r.status_code == 429

    limiter.reset()
    r = client.get("/health", headers=headers)
    assert r.status_code == 200


def test_rate_limit_middleware_skipped_in_local_mode(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", "")
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "true")
    limiter.reset()
    client = TestClient(app)
    # Many requests should all pass in local mode.
    for _ in range(5):
        r = client.get("/health")
        assert r.status_code == 200, r.text

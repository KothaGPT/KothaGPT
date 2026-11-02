"""Tests for the web-client-facing API surface (auth, datasets, evaluations, training, usage)."""

import pytest
from fastapi.testclient import TestClient

from services.api.app import app
from services.api.rate_limit import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch):
    """Ensure each test starts with a clean rate-limiter bucket and no leaked
    API token. Other modules mutate the shared limiter's ``max_requests``/
    ``window`` (e.g. test_rate_limit_middleware), so restore the defaults too."""
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "true")
    limiter.max_requests = 60
    limiter.window = 60
    limiter.reset()
    yield
    limiter.max_requests = 60
    limiter.window = 60
    limiter.reset()


def _env(monkeypatch, token: str = "tok123"):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", token)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "false")


def test_auth_requires_token(monkeypatch):
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "false")
    r = client.get("/v1/me")
    assert r.status_code == 503


def test_me_and_login(monkeypatch):
    _env(monkeypatch)
    r = client.get("/v1/me", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401

    r = client.get("/v1/me", headers={"Authorization": "Bearer tok123"})
    assert r.status_code == 200
    assert r.json()["id"] == "dev-user-1"

    r = client.post("/v1/auth/login", headers={"Authorization": "Bearer tok123"})
    assert r.status_code == 200
    body = r.json()
    assert "accessToken" in body
    assert "user" in body
    assert body["user"]["id"] == "dev-user-1"


def test_logout(monkeypatch):
    _env(monkeypatch)
    r = client.put("/v1/auth/logout")
    assert r.status_code == 200


def test_datasets_list_get_preview():
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["data"]) == 2

    r = client.get("/v1/datasets/ds-bangla-corpus")
    assert r.status_code == 200
    assert r.json()["name"] == "Bangla General Corpus"

    r = client.get("/v1/datasets/ds-bangla-corpus/preview")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 2

    r = client.get("/v1/datasets/ds-bangla-corpus/versions")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_datasets_delete_forbidden():
    r = client.delete("/v1/datasets/ds-bangla-corpus")
    assert r.status_code == 403


def test_benchmarks_and_evaluations(monkeypatch):
    _env(monkeypatch)
    r = client.get("/v1/benchmarks")
    assert r.status_code == 200
    assert len(r.json()) == 4

    r = client.post(
        "/v1/evaluations",
        json={"name": "my-eval", "model": "kothagpt", "benchmark": "bangla-qa"},
    )
    assert r.status_code == 201
    eid = r.json()["id"]

    r = client.get(f"/v1/evaluations/{eid}")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"

    r = client.get("/v1/evaluations")
    assert r.status_code == 200
    assert r.json()["total"] >= 1

    r = client.delete(f"/v1/evaluations/{eid}")
    assert r.status_code == 204


def test_training_lifecycle(monkeypatch):
    _env(monkeypatch)
    r = client.get("/v1/training")
    assert r.status_code == 200

    r = client.post(
        "/v1/training",
        json={
            "name": "t1",
            "model": "kothagpt",
            "dataset": "ds-1",
            "tokenizer": "tok-1",
            "config": {},
        },
    )
    assert r.status_code == 201
    tid = r.json()["id"]

    r = client.get(f"/v1/training/{tid}")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"

    r = client.patch(f"/v1/training/{tid}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

    r = client.patch(f"/v1/training/{tid}/pause")
    assert r.status_code == 200
    assert r.json()["status"] == "paused"

    r = client.patch(f"/v1/training/{tid}/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"

    r = client.get(f"/v1/training/{tid}/checkpoints")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.get(f"/v1/training/{tid}/logs")
    assert r.status_code == 200


def test_usage(monkeypatch):
    _env(monkeypatch)
    r = client.get("/v1/usage/summary")
    assert r.status_code == 200
    assert r.json()["requests"] == 12340

    r = client.get("/v1/usage/timeseries")
    assert r.status_code == 200
    assert len(r.json()) == 7

    r = client.get("/v1/usage/by-model")
    assert r.status_code == 200
    assert "kothagpt" in r.json()


def test_training_404():
    r = client.get("/v1/training/does-not-exist")
    assert r.status_code == 404


def test_evaluation_404():
    r = client.get("/v1/evaluations/does-not-exist")
    assert r.status_code == 404

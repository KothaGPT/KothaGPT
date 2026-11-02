"""Agent/run state must survive an instance restart.

On ephemeral hosts (e.g. Vercel serverless functions) each request may hit a
fresh instance, so ``MockBackend`` persists agents and runs through
``REDIS_URL``/``DATABASE_URL``-backed stores instead of process memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
import pytest

from services.api.api.schemas import AgentSpec
from services.api.core.agent_store import (
    MemoryStore,
    RedisStore,
    SqlStore,
    create_store_from_env,
)
from services.api.core.mock_backend import MockBackend

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_PORT = 8017
API_TEST_TOKEN = "test-token-abc123"
AUTH_HEADERS = {"Authorization": f"Bearer {API_TEST_TOKEN}"}


def _start_server(database_url: str):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.api.app:app",
            "--port",
            str(API_PORT),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "KOTHAGPT_API_TOKEN": API_TEST_TOKEN,
            # Point the deployment's agent state at a durable store.
            "DATABASE_URL": database_url,
        },
    )
    for _ in range(50):
        try:
            if httpx.get(f"http://localhost:{API_PORT}/health").status_code == 200:
                return proc, f"http://localhost:{API_PORT}"
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("server did not start")


def test_agent_and_run_survive_instance_restart(tmp_path) -> None:
    """Create an agent, restart the instance, then retrieve it again."""
    database_url = f"sqlite:///{tmp_path / 'agent-state.db'}"

    proc, base = _start_server(database_url)
    try:
        created = httpx.post(
            f"{base}/v1/agents",
            json={"name": "durable", "instructions": "Persist across restarts."},
            headers=AUTH_HEADERS,
        )
        assert created.status_code == 201, created.text
        agent = created.json()
        run_response = httpx.post(
            f"{base}/v1/agents/{agent['id']}/runs",
            json={"message": "hello"},
            headers=AUTH_HEADERS,
        )
        assert run_response.status_code == 201, run_response.text
        run = run_response.json()
    finally:
        proc.terminate()  # Simulate the instance being recycled/redeployed.
        proc.wait(timeout=10)

    proc, base = _start_server(database_url)  # A brand-new instance.
    try:
        fetched = httpx.get(f"{base}/v1/agents/{agent['id']}", headers=AUTH_HEADERS)
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["name"] == "durable"
        assert fetched.json()["instructions"] == "Persist across restarts."

        run_fetched = httpx.get(
            f"{base}/v1/agents/{agent['id']}/runs/{run['id']}", headers=AUTH_HEADERS
        )
        assert run_fetched.status_code == 200, run_fetched.text
        assert run_fetched.json()["output"] == run["output"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_redis_store_roundtrip_across_backend_instances(monkeypatch) -> None:
    fakeredis = pytest.importorskip("fakeredis")
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("redis.Redis.from_url", classmethod(lambda cls, *args, **kwargs: fake))

    url = "redis://localhost:6390/0"
    first = MockBackend(store=create_store_from_env({"REDIS_URL": url}))
    agent = first.create_agent(AgentSpec(name="durable", instructions="remember me"))
    run = first.run_agent(agent.id, "hello")

    second = MockBackend(store=create_store_from_env({"REDIS_URL": url}))
    assert second.get_agent(agent.id).instructions == "remember me"
    assert second.get_run(run.id).output == run.output
    assert [a.name for a in second.list_agents()] == ["durable"]

    second.delete_agent(agent.id)
    with pytest.raises(KeyError):
        first.get_agent(agent.id)


def test_store_selection_prefers_redis_then_sql_then_memory() -> None:
    assert isinstance(create_store_from_env({}), MemoryStore)
    assert isinstance(create_store_from_env({"DATABASE_URL": "sqlite:///state.db"}), SqlStore)
    assert isinstance(create_store_from_env({"REDIS_URL": "redis://localhost/0"}), RedisStore)
    assert isinstance(
        create_store_from_env(
            {"REDIS_URL": "redis://localhost/0", "DATABASE_URL": "sqlite:///state.db"}
        ),
        RedisStore,
    )

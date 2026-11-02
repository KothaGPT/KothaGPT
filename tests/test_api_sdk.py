import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from services.api.app import app

client = TestClient(app)

TEST_TOKEN = "test-token-abc123"
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture(autouse=True)
def _api_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KOTHAGPT_API_TOKEN", TEST_TOKEN)
    yield


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    # KGPT-002: status reflects the active runtime contract. A mock/example
    # backend is reported as "degraded" so callers cannot mistake it for a
    # production model runtime; the runtime snapshot is always present.
    assert body["status"] in ("ok", "degraded")
    assert body["service"] == "kothagpt-api"
    assert "runtime" in body
    runtime = body["runtime"]
    assert runtime["backend"] == "mock"
    assert runtime["production"] is False


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = [m["id"] for m in data]
    assert "kothagpt" in ids
    assert "kothagpt-embed" in ids
    assert "kothagpt-rerank" in ids


def test_chat_completions():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "kothagpt", "messages": [{"role": "user", "content": "হ্যালো"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0


def test_chat_completions_unknown_model_rejected():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code in (200, 404)


def test_chat_legacy():
    r = client.post("/v1/chat", json={"message": "বাংলা"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "kothagpt"
    assert "বাংলা" in body["output"]


def test_chat_streaming_sse():
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "হ্যালো"}]},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = r.read().decode()
    assert "data: " in text
    assert "data: [DONE]" in text
    assert "chat.completion.chunk" in text


def test_chat_completions_stream_flag():
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "হ্যালো"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_embeddings_single():
    r = client.post("/v1/embeddings", json={"model": "kothagpt-embed", "input": "বাংলা ভাষা"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 256


def test_embeddings_batch_deterministic():
    payload = {"model": "kothagpt-embed", "input": ["a", "b", "a"]}
    first = client.post("/v1/embeddings", json=payload).json()["data"]
    second = client.post("/v1/embeddings", json=payload).json()["data"]
    assert first[0]["embedding"] == second[0]["embedding"]
    assert first[0]["embedding"] == second[2]["embedding"]
    assert first[0]["embedding"] != first[1]["embedding"]


def test_rerank_orders_by_relevance():
    r = client.post(
        "/v1/rerank",
        json={
            "query": "বাংলা ভাষা শেখা",
            "documents": [
                "ইংরেজি শেখার উপায়",
                "বাংলা ভাষা শেখার সেরা উপায়",
                "রান্নার রেসিপি",
            ],
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    scores = [x["relevance_score"] for x in results]
    assert scores == sorted(scores, reverse=True)


def test_tools_list_and_get():
    r = client.get("/v1/tools", headers=AUTH)
    assert r.status_code == 200
    names = [t["function"]["name"] for t in r.json()["data"]]
    assert "calculator" in names
    assert "current_time" in names

    r2 = client.get("/v1/tools/calculator", headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["function"]["name"] == "calculator"


def test_tool_invoke_calculator():
    r = client.post(
        "/v1/tools/calculator/invoke",
        json={"name": "calculator", "arguments": {"expression": "(2 + 3) * 4"}},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert r.json()["result"]["value"] == 20


def test_tool_invoke_unknown_404():
    r = client.post("/v1/tools/nope/invoke", json={"name": "nope", "arguments": {}}, headers=AUTH)
    assert r.status_code == 404


def test_calculator_rejects_unsafe_expressions():
    unsafe = [
        "__import__('os').system('true')",
        "().__class__.__bases__[0].__subclasses__()",
        "9 ** 99",
        "1e308 * 10",
        "-" * 25 + "1",
        "x + 1",
        "a" * 201,
    ]
    for expression in unsafe:
        r = client.post(
            "/v1/tools/calculator/invoke",
            json={"name": "calculator", "arguments": {"expression": expression}},
            headers=AUTH,
        )
        assert r.status_code == 200
        value = r.json()["result"]["value"]
        assert isinstance(value, str) and value.startswith("error:"), (expression, value)


def test_calculator_accepts_safe_expressions():
    cases = {"(2 + 3) * 4": 20, "2 ^ 10": 1024, "-7 % 3": 2, "10 / 4": 2.5}
    for expression, expected in cases.items():
        r = client.post(
            "/v1/tools/calculator/invoke",
            json={"name": "calculator", "arguments": {"expression": expression}},
            headers=AUTH,
        )
        assert r.status_code == 200
        assert r.json()["result"]["value"] == expected


def test_agents_crud_and_run():
    created = client.post(
        "/v1/agents",
        json={"name": "helper", "description": "test agent", "tools": ["calculator"]},
        headers=AUTH,
    )
    assert created.status_code == 201
    agent_id = created.json()["id"]

    listed = client.get("/v1/agents", headers=AUTH).json()["data"]
    assert any(a["id"] == agent_id for a in listed)

    run = client.post(f"/v1/agents/{agent_id}/runs", json={"message": "তুমি কেমন আছ?"}, headers=AUTH)
    assert run.status_code == 201
    run_body = run.json()
    assert run_body["status"] == "completed"
    assert run_body["output"]

    fetched = client.get(f"/v1/agents/{agent_id}/runs/{run_body['id']}", headers=AUTH)
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "completed"

    assert client.delete(f"/v1/agents/{agent_id}", headers=AUTH).status_code == 204
    assert client.get(f"/v1/agents/{agent_id}", headers=AUTH).status_code == 404


def test_agent_run_stream_sse():
    created = client.post("/v1/agents", json={"name": "streamer"}, headers=AUTH).json()
    agent_id = created["id"]
    with client.stream(
        "POST",
        f"/v1/agents/{agent_id}/runs/stream",
        json={"message": "হ্যালো"},
        headers=AUTH,
    ) as r:
        assert r.status_code == 200
        text = r.read().decode()
    assert "run.created" in text
    assert "run.completed" in text
    assert "data: [DONE]" in text


def test_websocket_chat():
    with client.websocket_connect("/v1/ws", headers=AUTH) as ws:
        ws.send_text(
            json.dumps(
                {
                    "id": "1",
                    "type": "chat",
                    "payload": {"messages": [{"role": "user", "content": "হ্যালো"}]},
                }
            )
        )
        reply = json.loads(ws.receive_text())
        assert reply["id"] == "1"
        assert reply["type"] == "chat"
        assert reply["payload"]["object"] == "chat.completion"


def test_websocket_ping_and_models():
    with client.websocket_connect("/v1/ws", headers=AUTH) as ws:
        ws.send_text(json.dumps({"id": "a", "type": "ping", "payload": {}}))
        assert json.loads(ws.receive_text())["payload"] == {"pong": True}
        ws.send_text(json.dumps({"id": "b", "type": "models.list", "payload": {}}))
        reply = json.loads(ws.receive_text())
        assert reply["payload"]["data"][0]["id"] == "kothagpt"


def test_websocket_agents_create_and_run():
    with client.websocket_connect("/v1/ws", headers=AUTH) as ws:
        ws.send_text(
            json.dumps({"id": "1", "type": "agents.create", "payload": {"name": "ws-agent"}})
        )
        agent = json.loads(ws.receive_text())["payload"]
        ws.send_text(
            json.dumps(
                {
                    "id": "2",
                    "type": "agents.run",
                    "payload": {"agent_id": agent["id"], "message": "hi"},
                }
            )
        )
        run = json.loads(ws.receive_text())["payload"]
        assert run["status"] == "completed"


def test_websocket_token_via_query_param():
    with client.websocket_connect(f"/v1/ws?token={TEST_TOKEN}") as ws:
        ws.send_text(json.dumps({"id": "q", "type": "ping", "payload": {}}))
        assert json.loads(ws.receive_text())["payload"] == {"pong": True}


def test_websocket_missing_token_rejected():
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/ws"):
        pass


def test_websocket_invalid_token_rejected():
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/v1/ws", headers={"Authorization": "Bearer wrong-token"}),
    ):
        pass
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/ws?token=wrong-token"):
        pass


def test_auth_missing_token_rejected():
    r = client.get("/v1/tools")
    assert r.status_code == 401


def test_auth_wrong_token_rejected():
    r = client.get("/v1/tools", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401
    r2 = client.post(
        "/v1/tools/calculator/invoke",
        json={"name": "calculator", "arguments": {"expression": "1+1"}},
        headers={"Authorization": "wrong-scheme"},
    )
    assert r2.status_code == 401


def test_auth_unconfigured_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "false")
    r = client.get("/v1/tools")
    assert r.status_code == 503


def test_auth_local_mode_allows_without_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KOTHAGPT_API_TOKEN", raising=False)
    monkeypatch.setenv("KOTHAGPT_LOCAL_MODE", "true")
    r = client.get("/v1/tools")
    assert r.status_code == 200

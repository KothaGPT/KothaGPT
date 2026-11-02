"""Tests for the conversations API (``/v1/conversations``)."""

from fastapi.testclient import TestClient

from services.api.app import app

client = TestClient(app)


def _create(title: str = "Test chat", model: str = "kothagpt") -> str:
    r = client.post("/v1/conversations", json={"title": title, "model": model})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_create_conversation():
    cid = _create("আমার কথোপকথন", "kothagpt-small")
    r = client.get(f"/v1/conversations/{cid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == cid
    assert body["title"] == "আমার কথোপকথন"
    assert body["model"] == "kothagpt-small"
    assert "updatedAt" in body


def test_list_conversations():
    _create("first")
    _create("second")
    r = client.get("/v1/conversations")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert len(body["data"]) >= 2


def test_messages_round_trip():
    cid = _create()
    # Empty initially
    r = client.get(f"/v1/conversations/{cid}/messages")
    assert r.status_code == 200
    assert r.json() == []


def test_delete_conversation():
    cid = _create()
    r = client.delete(f"/v1/conversations/{cid}")
    assert r.status_code == 204
    r = client.get(f"/v1/conversations/{cid}")
    assert r.status_code == 404


def test_chat_stores_messages_in_conversation():
    cid = _create()
    r = client.post(
        "/v1/chat",
        json={
            "conversationId": cid,
            "model": "kothagpt",
            "messages": [{"role": "user", "content": "বাংলায় বলো কিছু"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    text = r.text
    assert "start" in text and "delta" in text and "done" in text

    r = client.get(f"/v1/conversations/{cid}/messages")
    assert r.status_code == 200
    msgs = r.json()
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "assistant" in roles


def test_chat_legacy_still_works():
    r = client.post("/v1/chat", json={"message": "হ্যালো", "model": "kothagpt"})
    assert r.status_code == 200
    assert r.json()["output"]


def test_get_nonexistent_conversation_404():
    r = client.get("/v1/conversations/does-not-exist")
    assert r.status_code == 404


def test_delete_nonexistent_conversation_404():
    r = client.delete("/v1/conversations/does-not-exist")
    assert r.status_code == 404

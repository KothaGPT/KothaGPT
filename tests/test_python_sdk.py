import asyncio

import pytest
from kothagpt import AsyncKothaGPT, KothaGPT
from kothagpt.websocket import WebSocketClient


@pytest.fixture(scope="module")
def client(server):
    from tests.conftest import API_TEST_TOKEN

    with KothaGPT(base_url=server, api_key=API_TEST_TOKEN) as c:
        yield c


def test_list_models(client):
    models = client.models.list()
    assert [m.id for m in models] == [
        "kothagpt-nano",
        "kothagpt-small",
        "kothagpt-medium",
        "kothagpt-large",
        "kothagpt",
        "kothagpt-embed",
        "kothagpt-rerank",
    ]


def test_chat(client):
    resp = client.chat.create(messages=[{"role": "user", "content": "হ্যালো"}])
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert resp.text
    assert resp.usage.total_tokens > 0


def test_chat_stream(client):
    chunks = list(client.chat.stream(messages=[{"role": "user", "content": "হ্যালো"}]))
    assert chunks
    text = "".join(c.delta for c in chunks)
    assert text
    assert any(c.choices[0].get("finish_reason") == "stop" for c in chunks)


def test_embeddings(client):
    resp = client.embeddings.create(["বাংলা", "ভাষা"])
    assert len(resp.data) == 2
    assert len(resp.data[0].embedding) == 256


def test_rerank(client):
    resp = client.rerank.create("বাংলা ভাষা", ["অন্য কিছু", "বাংলা ভাষা শেখা"])
    assert resp.results[0].index in (0, 1)


def test_tools(client):
    names = [t.function.name for t in client.tools.list()]
    assert "calculator" in names
    result = client.tools.invoke("calculator", {"expression": "2 + 3 * 4"})
    assert result["value"] == 14


def test_agents(client):
    agent = client.agents.create({"name": "sdk-agent", "tools": ["calculator"]})
    assert agent.id
    run = client.agents.run(agent.id, "হাই")
    assert run.status == "completed"
    assert run.output
    fetched = client.agents.get_run(agent.id, run.id)
    assert fetched.id == run.id
    assert [a.id for a in client.agents.list()] == [agent.id]
    client.agents.delete(agent.id)


def test_agent_stream(client):
    agent = client.agents.create({"name": "sdk-streamer"})
    events = list(client.agents.stream(agent.id, "হ্যালো"))
    assert any(e["event"] == "run.created" for e in events)
    assert any(e["event"] == "run.completed" for e in events)


def test_errors(client):
    with pytest.raises(Exception) as excinfo:
        client.tools.invoke("nope", {})
    assert excinfo.value.status_code == 404


def test_async_client(server):
    from tests.conftest import API_TEST_TOKEN

    async def main():
        async with AsyncKothaGPT(base_url=server, api_key=API_TEST_TOKEN) as c:
            resp = await c.chat.create(messages=[{"role": "user", "content": "হাই"}])
            assert resp.text
            emb = await c.embeddings.create("বাংলা")
            assert len(emb.data[0].embedding) == 256
            chunks = []
            async for chunk in c.chat.stream(messages=[{"role": "user", "content": "হাই"}]):
                chunks.append(chunk)
            assert chunks
            agent = await c.agents.create({"name": "async-agent"})
            run = await c.agents.run(agent.id, "হাই")
            assert run.status == "completed"

    asyncio.run(main())


def test_websocket(server):
    from tests.conftest import API_TEST_TOKEN

    async def main():
        ws_url = server.replace("http", "ws")
        async with WebSocketClient(base_url=ws_url, api_key=API_TEST_TOKEN) as ws:
            completion = await ws.chat([{"role": "user", "content": "হ্যালো"}])
            assert completion.text
            emb = await ws.embed("বাংলা")
            assert len(emb.data[0].embedding) == 256
            tools = await ws.tools_list()
            assert any(t["function"]["name"] == "calculator" for t in tools)
            agent = await ws.agents_create({"name": "ws-agent"})
            run = await ws.agents_run(agent.id, "হাই")
            assert run.status == "completed"

    asyncio.run(main())

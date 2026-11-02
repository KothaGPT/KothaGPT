# Kotha GPT Python SDK

Official Python client for the [Kotha GPT](https://github.com/khulnasoft/KothaGPT) API.

## Install

```bash
pip install kothagpt
```

Requires Python 3.11+. For WebSocket support the `websockets` package is used
(it is installed as a dependency).

## Quick start

```python
from kothagpt import KothaGPT

client = KothaGPT(base_url="http://localhost:8000", api_key="sk-...")

completion = client.chat.create(
    messages=[{"role": "user", "content": "বাংলায় একটি ছোট গল্প বলো"}],
)
print(completion.text)
```

## Streaming

```python
for chunk in client.chat.stream(
    messages=[{"role": "user", "content": "একটি গল্প বলো"}],
):
    print(chunk.delta, end="", flush=True)
print()
```

## Embeddings

```python
response = client.embeddings.create(["বাংলা", "বাংলাদেশ", "ভাষা"])
for item in response.data:
    print(item.index, len(item.embedding))
```

## Reranking

```python
response = client.rerank.create(
    query="বাংলা ভাষা শেখার সেরা উপায়",
    documents=["রান্নার রেসিপি", "বাংলা ব্যাকরণের বই", "ইংরেজি শেখার কোর্স"],
    top_n=2,
)
for result in response.results:
    print(result.index, result.relevance_score, result.document)
```

## Tools

```python
tools = client.tools.list()
result = client.tools.invoke("calculator", {"expression": "(2 + 3) * 4"})
print(result)
```

## Agents

```python
agent = client.agents.create(
    {
        "name": "research-assistant",
        "instructions": "সংক্ষিপ্ত উত্তর দাও।",
        "tools": ["web_search"],
    }
)

run = client.agents.run(agent.id, "গত বছরের বাজার সম্পর্কে বলো")
print(run.output)
```

## WebSocket

```python
import asyncio
from kothagpt.websocket import WebSocketClient


async def main():
    async with WebSocketClient(base_url="ws://localhost:8000") as ws:
        completion = await ws.chat([{"role": "user", "content": "হ্যালো"}])
        print(completion.text)


asyncio.run(main())
```

## Async client

```python
import asyncio
from kothagpt import AsyncKothaGPT


async def main():
    async with AsyncKothaGPT() as client:
        completion = await client.chat.create(messages=[{"role": "user", "content": "হ্যালো"}])
        print(completion.text)


asyncio.run(main())
```

## Models

```python
for model in client.models.list():
    print(model.id, model.context_window)
```

## Configuration

| Environment variable | Used for            |
| -------------------- | ------------------- |
| `KOTHAGPT_API_URL`   | API base URL        |
| `KOTHAGPT_API_KEY`   | Bearer API key      |

Errors raised by the SDK subclass `KothaGPTError` (`APIError`, `AuthenticationError`,
`NotFoundError`, ...) so callers can catch them by type.
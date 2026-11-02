"""Chat over the WebSocket endpoint.

Usage:
    python examples/python/websocket.py
"""

import asyncio

from kothagpt.websocket import WebSocketClient


async def main() -> None:
    async with WebSocketClient(base_url="ws://localhost:8000") as ws:
        completion = await ws.chat([{"role": "user", "content": "হ্যালো, কেমন আছো?"}])
        print(completion.text)

        embedded = await ws.embed("বাংলা ভাষা")
        print(f"\nembedding dim: {len(embedded.data[0].embedding)}")


if __name__ == "__main__":
    asyncio.run(main())

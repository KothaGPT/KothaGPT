"""Streaming chat with the Python SDK (SSE).

Usage:
    python examples/python/streaming.py "দুই লাইনে একটি কবিতা লেখো"
"""

import sys

from kothagpt import KothaGPT


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "একটি ছোট গল্প বলো"
    with KothaGPT() as client:
        print("Streaming response:\n")
        for chunk in client.chat.stream(
            messages=[{"role": "user", "content": message}],
        ):
            print(chunk.delta, end="", flush=True)
        print()


if __name__ == "__main__":
    main()

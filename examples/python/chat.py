"""Basic chat with the Python SDK.

Usage:
    python examples/python/chat.py "বাংলায় একটি ছোট গল্প বলো"
"""

import sys

from kothagpt import KothaGPT


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "বাংলায় হ্যালো বলো"
    with KothaGPT() as client:
        completion = client.chat.create(
            messages=[{"role": "user", "content": message}],
            model="kothagpt",
        )
        print(completion.text)
        print(f"\n[usage: {completion.usage.total_tokens} tokens]")


if __name__ == "__main__":
    main()

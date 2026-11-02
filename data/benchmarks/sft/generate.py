"""Generate deterministic, per-family SFT benchmark JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

_DEFAULT_OUT = Path(__file__).parent / "v1"
_LANGUAGES = {"bn", "en", "multilingual", "mixed", "unknown"}
_CATEGORIES = {
    "instruction",
    "coding",
    "reasoning",
    "conversation",
    "function_call",
    "tool_use",
}
_ROLES = {"system", "user", "assistant", "tool"}

_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "sft-bn": [
        {
            "instruction": "২ + ৩ কত? শুধু উত্তর দিন।",
            "output": "৫",
            "language": "bn",
            "category": "instruction",
        },
        {
            "instruction": "বাংলাদেশের রাজধানীর নাম কী?",
            "output": "বাংলাদেশের রাজধানী ঢাকা।",
            "language": "bn",
            "category": "instruction",
        },
        {
            "instruction": "একটি বাক্যে বলুন, পানি ০° সেলসিয়াসে কী অবস্থায় থাকে।",
            "output": "০° সেলসিয়াসে পানি জমে বরফ হয়।",
            "language": "bn",
            "category": "instruction",
        },
        {
            "instruction": "পাঁচটি আম থেকে দুটি আম দিলে কয়টি আম থাকে?",
            "output": "পাঁচটি আম থেকে দুটি দিলে তিনটি আম থাকে।",
            "language": "bn",
            "category": "instruction",
        },
    ],
    "sft-en": [
        {
            "instruction": "What is 12 plus 8? Answer with the number only.",
            "output": "20",
            "language": "en",
            "category": "instruction",
        },
        {
            "instruction": "Name the capital of Japan.",
            "output": "Tokyo.",
            "language": "en",
            "category": "instruction",
        },
        {
            "instruction": "Rewrite in the past tense: She walks to school.",
            "output": "She walked to school.",
            "language": "en",
            "category": "instruction",
        },
        {
            "instruction": "Give two examples of renewable energy.",
            "output": "Solar energy and wind energy.",
            "language": "en",
            "category": "instruction",
        },
    ],
    "sft-multilingual": [
        {
            "instruction": "বাংলা ও ইংরেজিতে 'ধন্যবাদ' বলুন।",
            "output": "বাংলা: ধন্যবাদ। English: Thank you.",
            "language": "multilingual",
            "category": "instruction",
        },
        {
            "instruction": "Translate 'Good morning' into Bengali and include the English phrase.",
            "output": "Good morning means সুপ্রভাত.",
            "language": "multilingual",
            "category": "instruction",
        },
        {
            "instruction": "What is 4 + 5? উত্তরটি বাংলাতেও লিখুন।",
            "output": "9. বাংলায়: নয়।",
            "language": "multilingual",
            "category": "instruction",
        },
        {
            "instruction": "বাংলায় 'water' শব্দের অর্থ কী? Give a short answer in both languages.",
            "output": "Water means পানি (জল).",
            "language": "multilingual",
            "category": "instruction",
        },
    ],
    "sft-code": [
        {
            "instruction": "Write a Python function named `add` that returns the sum of two arguments a and b.",
            "output": "def add(a, b):\n    return a + b",
            "language": "en",
            "category": "coding",
        },
        {
            "instruction": "Write a Python function `is_even(n)` that returns True when n is even.",
            "output": "def is_even(n):\n    return n % 2 == 0",
            "language": "en",
            "category": "coding",
        },
        {
            "instruction": "Write a Python function `square(n)` that returns the square of n.",
            "output": "def square(n):\n    return n * n",
            "language": "en",
            "category": "coding",
        },
        {
            "instruction": "Write a Python function `first(items)` that returns the first item, or None for an empty list.",
            "output": "def first(items):\n    return items[0] if items else None",
            "language": "en",
            "category": "coding",
        },
    ],
    "sft-reasoning": [
        {
            "instruction": "A box has 3 rows of 4 oranges. How many oranges are there? Show one calculation.",
            "output": "There are 3 × 4 = 12 oranges.",
            "language": "en",
            "category": "reasoning",
        },
        {
            "instruction": "রিমার কাছে ১০টি কলম ছিল। সে ৩টি দিল। তার কাছে কয়টি রইল? হিসাব দেখান।",
            "output": "১০ - ৩ = ৭। রিমার কাছে ৭টি কলম রইল।",
            "language": "bn",
            "category": "reasoning",
        },
        {
            "instruction": "A meeting starts at 2:00 PM and lasts 90 minutes. When does it end?",
            "output": "It ends at 3:30 PM.",
            "language": "en",
            "category": "reasoning",
        },
        {
            "instruction": "সোমবারের দুই দিন পর কোন দিন?",
            "output": "সোমবারের দুই দিন পর বুধবার।",
            "language": "bn",
            "category": "reasoning",
        },
    ],
    "sft-conversation": [
        {
            "messages": [
                {"role": "user", "content": "I want to start reading more."},
                {"role": "assistant", "content": "What kind of books do you enjoy?"},
                {"role": "user", "content": "Short science books."},
            ],
            "output": "Try a short popular-science book and read a few pages each day.",
            "language": "en",
            "category": "conversation",
        },
        {
            "messages": [
                {"role": "user", "content": "আজ খুব গরম।"},
                {"role": "assistant", "content": "পানি পান করছেন তো?"},
                {"role": "user", "content": "হ্যাঁ, তবে বাইরে যেতে হবে।"},
            ],
            "output": "বাইরে গেলে ছাতা বা টুপি নিন, আর সম্ভব হলে ছায়ায় বিশ্রাম করুন।",
            "language": "bn",
            "category": "conversation",
        },
        {
            "messages": [
                {"role": "user", "content": "Can you help me plan dinner?"},
                {"role": "assistant", "content": "Sure. What ingredients do you have?"},
                {"role": "user", "content": "Rice, eggs, and spinach."},
            ],
            "output": "You could make spinach egg fried rice with those ingredients.",
            "language": "en",
            "category": "conversation",
        },
        {
            "messages": [
                {"role": "user", "content": "আমি নতুন করে বাংলা শিখছি।"},
                {"role": "assistant", "content": "দারুণ! কোন বিষয় দিয়ে শুরু করতে চান?"},
                {"role": "user", "content": "প্রতিদিনের সহজ বাক্য।"},
            ],
            "output": "শুরু করতে পারেন: 'আপনি কেমন আছেন?' এবং 'ধন্যবাদ' দিয়ে।",
            "language": "bn",
            "category": "conversation",
        },
    ],
    "sft-fc": [
        {
            "instruction": "Call `add` with a=2 and b=3.",
            "output": '{"name":"add","arguments":{"a":2,"b":3}}',
            "language": "en",
            "category": "function_call",
            "function_call": {"name": "add", "arguments": {"a": 2, "b": 3}},
        },
        {
            "instruction": "ঢাকার আবহাওয়া জানতে `get_weather` কল করুন।",
            "output": '{"name":"get_weather","arguments":{"city":"Dhaka"}}',
            "language": "bn",
            "category": "function_call",
            "function_call": {"name": "get_weather", "arguments": {"city": "Dhaka"}},
        },
        {
            "instruction": "Search for the phrase KothaGPT in English.",
            "output": '{"name":"search","arguments":{"query":"KothaGPT","language":"en"}}',
            "language": "en",
            "category": "function_call",
            "function_call": {
                "name": "search",
                "arguments": {"query": "KothaGPT", "language": "en"},
            },
        },
        {
            "instruction": "`multiply` ফাংশন দিয়ে ৬ ও ৭ গুণ করুন।",
            "output": '{"name":"multiply","arguments":{"a":6,"b":7}}',
            "language": "bn",
            "category": "function_call",
            "function_call": {"name": "multiply", "arguments": {"a": 6, "b": 7}},
        },
    ],
    "sft-tool-use": [
        {
            "instruction": "Use the calculator tool to compute 9 + 6.",
            "output": '{"tool":"calculator","arguments":{"expression":"9 + 6"}}',
            "language": "en",
            "category": "tool_use",
            "tools": [{"name": "calculator", "arguments": {"expression": "string"}}],
        },
        {
            "instruction": "Search the web for KothaGPT documentation.",
            "output": '{"tool":"web_search","arguments":{"query":"KothaGPT documentation"}}',
            "language": "en",
            "category": "tool_use",
            "tools": [{"name": "web_search", "arguments": {"query": "string"}}],
        },
        {
            "instruction": "বাংলাদেশের রাজধানী জানতে knowledge base ব্যবহার করুন।",
            "output": '{"tool":"knowledge_base","arguments":{"query":"বাংলাদেশের রাজধানী"}}',
            "language": "bn",
            "category": "tool_use",
            "tools": [{"name": "knowledge_base", "arguments": {"query": "string"}}],
        },
        {
            "instruction": "Use the calculator tool to find 15 divided by 3.",
            "output": '{"tool":"calculator","arguments":{"expression":"15 / 3"}}',
            "language": "en",
            "category": "tool_use",
            "tools": [{"name": "calculator", "arguments": {"expression": "string"}}],
        },
    ],
}


def _assign_splits(family: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record_ids = [f"{family}:{index:03d}" for index in range(1, len(records) + 1)]
    test_count = max(1, round(len(records) * 0.2))
    test_ids = set(
        sorted(record_ids, key=lambda value: hashlib.sha256(value.encode("utf-8")).digest())[
            :test_count
        ]
    )
    assigned = []
    for record_id, record in zip(record_ids, records, strict=True):
        item = dict(record)
        item["metadata"] = {
            **item.get("metadata", {}),
            "record_id": record_id,
            "family": family,
            "split": "test" if record_id in test_ids else "dev",
        }
        _validate_record(item)
        assigned.append(item)
    return assigned


def _validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record.get("instruction", ""), str):
        raise TypeError("instruction must be a string")
    if not isinstance(record.get("output"), str):
        raise TypeError("output must be a string")
    if not record["output"].strip():
        raise ValueError("output must be a non-empty string")
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be an object")
        role = message.get("role")
        if not isinstance(role, str):
            raise TypeError("message role must be a string")
        if role not in _ROLES:
            raise ValueError(f"unsupported message role: {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise TypeError("message content must be a string")
        if not message["content"].strip():
            raise ValueError("message content must be a non-empty string")
    if not messages and not record.get("instruction", "").strip():
        raise ValueError("instruction or messages is required")
    language = record.get("language", "unknown")
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    if language not in _LANGUAGES:
        raise ValueError(f"unsupported language: {record.get('language')!r}")
    category = record.get("category", "instruction")
    if not isinstance(category, str):
        raise TypeError("category must be a string")
    if category not in _CATEGORIES:
        raise ValueError(f"unsupported category: {record.get('category')!r}")
    tools = record.get("tools") or []
    if not isinstance(tools, list):
        raise TypeError("tools must be a list")
    if not all(isinstance(tool, dict) for tool in tools):
        raise TypeError("tools must contain only objects")
    if not isinstance(record.get("metadata"), dict):
        raise TypeError("metadata must be an object")


def generate_all(out_dir: str | Path = _DEFAULT_OUT) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": "v1",
        "schema": "InstructionRecord JSONL with record_id/family/split in metadata",
        "split_policy": "stable SHA-256 ranking within each family; approximately 80% dev and 20% test",
        "families": {},
    }
    for family, records in _FAMILIES.items():
        path = out_dir / f"{family}.jsonl"
        assigned = _assign_splits(family, records)
        content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in assigned)
        path.write_text(content, encoding="utf-8")
        split_counts = dict(sorted(Counter(item["metadata"]["split"] for item in assigned).items()))
        manifest["families"][path.name] = {
            "records": len(assigned),
            "split_counts": split_counts,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    args = parser.parse_args(argv)
    manifest = generate_all(args.out_dir)
    print(
        f"generated {sum(item['records'] for item in manifest['families'].values())} SFT benchmark records in {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

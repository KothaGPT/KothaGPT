"""Validated JSONL instruction data and completion-only SFT batching."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

_ALLOWED_LANGUAGES = {"bn", "en", "multilingual", "mixed", "unknown"}
_ALLOWED_CATEGORIES = {
    "instruction",
    "general",
    "coding",
    "reasoning",
    "conversation",
    "function_call",
    "tool_use",
}


def _text(value: Any, field_name: str, *, required: bool = False) -> str:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


@dataclass(frozen=True)
class InstructionRecord:
    instruction: str
    output: str
    input: str = ""
    language: str = "unknown"
    category: str = "instruction"
    messages: tuple[dict[str, str], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    function_call: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InstructionRecord:
        if not isinstance(raw, dict):
            raise TypeError("instruction record must be an object")
        messages_raw = raw.get("messages") or []
        if messages_raw and (
            not isinstance(messages_raw, list) or not all(isinstance(m, dict) for m in messages_raw)
        ):
            raise ValueError("messages must be a list of objects")
        messages = []
        for message in messages_raw:
            role = _text(message.get("role"), "messages.role", required=True)
            content = _text(message.get("content"), "messages.content", required=True)
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported message role: {role}")
            messages.append({"role": role, "content": content})
        instruction = _text(raw.get("instruction"), "instruction")
        output = _text(raw.get("output"), "output")
        if not messages and not instruction:
            raise ValueError("instruction or messages is required")
        if not output and messages:
            assistants = [m["content"] for m in messages if m["role"] == "assistant"]
            output = assistants[-1] if assistants else ""
        if not output:
            raise ValueError("output must not be empty")
        language = _text(raw.get("language"), "language") or "unknown"
        category = _text(raw.get("category"), "category") or "instruction"
        if language not in _ALLOWED_LANGUAGES:
            raise ValueError(f"unsupported language: {language}")
        if category not in _ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported category: {category}")
        tools = raw.get("tools") or []
        if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            raise ValueError("tools must be a list of objects")
        return cls(
            instruction,
            output,
            _text(raw.get("input"), "input"),
            language,
            category,
            tuple(messages),
            tuple(tools),
            raw.get("function_call"),
            raw.get("metadata") or {},
        )

    def prompt(self) -> str:
        if self.messages:
            return (
                "\n".join(
                    f"<{m['role']}>\n{m['content']}"
                    for m in self.messages
                    if m["role"] != "assistant"
                )
                + "\n<assistant>\n"
            )
        prompt = f"<user>\n{self.instruction}"
        if self.input:
            prompt += f"\n{self.input}"
        return prompt + "\n<assistant>\n"

    def completion(self) -> str:
        return self.output + "\n<eos>"


def load_jsonl(path: str | Path) -> list[InstructionRecord]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(InstructionRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid instruction record at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"no instruction records found in {path}")
    return records


def split_records(
    records: Iterable[InstructionRecord], validation_fraction: float = 0.1, seed: int = 0
) -> tuple[list[InstructionRecord], list[InstructionRecord]]:
    records = list(records)
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_validation = int(len(records) * validation_fraction)
    validation = {i for i in indices[:n_validation]}
    return [r for i, r in enumerate(records) if i not in validation], [
        r for i, r in enumerate(records) if i in validation
    ]


class InstructionDataset(Dataset):
    def __init__(self, records: Iterable[InstructionRecord], tokenizer, max_length: int):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prompt_ids = self.tokenizer.encode(record.prompt())
        completion_ids = self.tokenizer.encode(record.completion())
        ids = (prompt_ids + completion_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(ids))
        labels = [-100] * prompt_len + ids[prompt_len:]
        return {"input_ids": ids, "labels": labels, "record": record}


class InstructionCollator:
    def __init__(self, pad_id: int, max_length: int):
        self.pad_id, self.max_length = pad_id, max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = min(self.max_length, max(len(item["input_ids"]) for item in batch))
        input_ids, labels, attention = [], [], []
        for item in batch:
            ids, labs = item["input_ids"][:width], item["labels"][:width]
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_id] * padding)
            labels.append(labs + [-100] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "records": [item["record"] for item in batch],
        }

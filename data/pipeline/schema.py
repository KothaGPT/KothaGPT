"""Validation and canonicalization for instruction and conversation records."""

from __future__ import annotations

from typing import Any

__all__ = ["normalize_instruction_record"]

_ROLES = frozenset({"system", "user", "assistant", "tool"})
_CATEGORIES = frozenset(
    {"instruction", "general", "reasoning", "coding", "conversation", "function_call", "tool_use"}
)
_LANGUAGES = frozenset({"bn", "en", "multilingual", "mixed", "unknown"})


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def normalize_instruction_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a record and return a canonical messages-based representation.

    Alpaca-style ``instruction``/``input``/``output`` records are converted to
    user/assistant messages. Provenance is mandatory for every record.
    """
    if not isinstance(record, dict):
        raise TypeError("instruction record must be an object")

    messages = record.get("messages")
    if messages is None:
        instruction = _nonempty_string(record.get("instruction"), "instruction")
        output = _nonempty_string(record.get("output"), "output")
        extra_input = record.get("input", "")
        if not isinstance(extra_input, str):
            raise TypeError("input must be a string")
        user_content = "\n\n".join(part for part in (instruction, extra_input.strip()) if part)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    if len(messages) < 2:
        raise ValueError("messages must contain at least a user and assistant turn")

    normalized_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be an object")
        role = _nonempty_string(message.get("role"), f"messages[{index}].role")
        if role not in _ROLES:
            raise ValueError(f"messages[{index}].role is unsupported: {role!r}")
        content = _nonempty_string(message.get("content"), f"messages[{index}].content")
        normalized_messages.append({"role": role, "content": content})

    roles = [message["role"] for message in normalized_messages]
    if "user" not in roles:
        raise ValueError("messages must contain a user turn")
    if roles[-1] != "assistant":
        raise ValueError("messages must end with a non-empty assistant answer")

    source = _nonempty_string(record.get("source"), "source")
    license_id = _nonempty_string(record.get("license"), "license")
    category = record.get("category", "general")
    if not isinstance(category, str) or category not in _CATEGORIES:
        raise ValueError(f"category must be one of {sorted(_CATEGORIES)}")
    language = record.get("language", "unknown")
    if not isinstance(language, str) or language not in _LANGUAGES:
        raise ValueError(f"language must be one of {sorted(_LANGUAGES)}")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")

    normalized = {
        key: value
        for key, value in record.items()
        if key not in {"instruction", "input", "output", "messages", "quality_score", "split"}
    }
    normalized.update(
        messages=normalized_messages,
        source=source,
        license=license_id,
        category=category,
        language=language,
        metadata={key: value for key, value in metadata.items() if key != "split"},
    )
    return normalized

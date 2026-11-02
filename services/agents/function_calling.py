from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


def parse_function_call(value: str | dict[str, Any]) -> FunctionCall:
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise TypeError("function call requires a name")
    args = payload.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise TypeError("function arguments must be an object")
    return FunctionCall(payload["name"], args)

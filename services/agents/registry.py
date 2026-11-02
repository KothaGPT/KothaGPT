from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    permission: str = "read"
    enabled: bool = True


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"invalid or duplicate tool: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def list(self) -> list[ToolSpec]:
        return [
            spec
            for spec, _ in sorted(self._tools.values(), key=lambda item: item[0].name)
            if spec.enabled
        ]

    def invoke(
        self, name: str, arguments: dict[str, Any], *, allowed: set[str] | None = None
    ) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        spec, handler = self._tools[name]
        if not spec.enabled or (allowed is not None and name not in allowed):
            raise PermissionError(f"tool not permitted: {name}")
        required = spec.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")
        properties = spec.parameters.get("properties", {})
        unexpected = sorted(set(arguments) - set(properties)) if properties else []
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        return handler(**arguments)

"""packages/agents — action/tool registry and agent execution model.

Pure-stdlib, dependency-free agent primitives shared by the Python API and
first-class SDKs. A production deployment can supply its own ``decide``
callback (backed by a real model) while the registry, permission gate, and
execution loop remain portable.

Components:

- :class:`AgentSpec` / :class:`Agent` — declarative agent description.
- :class:`ToolRegistry` — named action/tool registry with permission gating.
- :class:`PermissionGate` — allow/approve policy for tool invocation.
- :class:`AgentEvent` — immutable event stream produced by a run.
- :func:`run_agent` — bounded step loop: decide → tool call → result → decide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Declaration of an invocable tool/action."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    permission: str = "read"
    enabled: bool = True


@dataclass(frozen=True)
class AgentSpec:
    """Declarative description of an agent."""

    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = field(default_factory=list)
    temperature: float | None = field(default=0.7)


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = field(default_factory=list)
    temperature: float | None = field(default=0.7)
    created_at: int = field(default_factory=lambda: __import__("time").time_ns() // 1_000_000_000)


@dataclass
class AgentRun:
    """Mutable record of a single agent execution."""

    id: str
    agent_id: str
    status: str = "queued"
    messages: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None
    created_at: int = field(default_factory=lambda: __import__("time").time_ns() // 1_000_000_000)
    updated_at: int = field(default_factory=lambda: __import__("time").time_ns() // 1_000_000_000)


@dataclass
class AgentEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class PermissionGate:
    """Allow/approve policy for tool invocation.

    ``allowed_tools``: tools that may be invoked at all.
    ``approvals``: tools that additionally require explicit approval when
    their risk is "high" (write/execute).
    """

    def __init__(
        self, allowed_tools: set[str] | None = None, approvals: set[str] | None = None
    ) -> None:
        self.allowed_tools: set[str] = set(allowed_tools or set())
        self.approvals: set[str] = set(approvals or set())

    def check(self, tool: str, *, risk: str = "low") -> None:
        if tool not in self.allowed_tools:
            raise PermissionError(f"tool denied: {tool}")
        if risk == "high" and tool not in self.approvals:
            raise PermissionError(f"approval required: {tool}")

    def approve(self, tool: str) -> None:
        if tool not in self.allowed_tools:
            raise PermissionError(f"cannot approve unallowed tool: {tool}")
        self.approvals.add(tool)


class ToolRegistry:
    """Named registry of actions/tools with handlers and permission specs."""

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

    def get(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

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


def _default_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def run_agent(
    message: str,
    *,
    decide: Callable[[str, list[AgentEvent]], str | None],
    registry: ToolRegistry,
    permissions: PermissionGate,
    max_steps: int = 8,
) -> tuple[str, list[AgentEvent]]:
    """Bounded agent execution loop.

    ``decide(prompt, events)`` returns either a tool-call string (parsed as a
    ``{"name": ..., "arguments": {...}}`` object) or ``None`` to finalize.
    Each iteration: decide → parse → permission check → invoke → append result
    → feed back as the next prompt. Stops on ``None``, error, or step limit.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    events = [AgentEvent("user", {"message": message})]
    prompt = message

    for _ in range(max_steps):
        output = decide(prompt, events)
        if output is None:
            return prompt, events

        events.append(AgentEvent("tool_call", {"output": output}))

        try:
            import json

            payload = json.loads(output) if isinstance(output, str) else output
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                raise TypeError("tool call requires a name")
            name = payload["name"]
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must be an object")

            spec = registry.get(name)
            if spec is None:
                raise KeyError(name)

            risk = "high" if spec.permission in {"write", "execute"} else "low"
            permissions.check(name, risk=risk)
            result = registry.invoke(name, arguments, allowed=permissions.allowed_tools)

            events.append(
                AgentEvent("tool_result", {"name": name, "arguments": arguments, "result": result})
            )
            prompt = str(result)

        except (TypeError, ValueError, KeyError, PermissionError) as exc:
            events.append(AgentEvent("error", {"error": str(exc)}))
            return "I could not safely execute that request.", events

    events.append(AgentEvent("limit", {"max_steps": max_steps}))
    return "The agent reached its step limit.", events


__all__ = [
    "Agent",
    "AgentEvent",
    "AgentRun",
    "AgentSpec",
    "PermissionGate",
    "ToolRegistry",
    "ToolSpec",
    "run_agent",
]

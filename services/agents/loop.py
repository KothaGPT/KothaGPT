from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .function_calling import parse_function_call
from .permissions import PermissionGate
from .registry import ToolRegistry


@dataclass
class AgentEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


def run_agent(
    message: str,
    *,
    decide: Callable[[str, list[AgentEvent]], str | None],
    registry: ToolRegistry,
    permissions: PermissionGate,
    max_steps: int = 8,
) -> tuple[str, list[AgentEvent]]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    events = [AgentEvent("user", {"message": message})]
    prompt = message
    for _ in range(max_steps):
        output = decide(prompt, events)
        if output is None:
            return prompt, events
        try:
            call = parse_function_call(output)
            events.append(AgentEvent("tool_call", {"name": call.name, "arguments": call.arguments}))
            spec = next((item for item in registry.list() if item.name == call.name), None)
            if spec is None:
                raise KeyError(call.name)
            permissions.check(
                call.name, risk="high" if spec.permission in {"write", "execute"} else "low"
            )
            result = registry.invoke(call.name, call.arguments, allowed=permissions.allowed_tools)
            events.append(AgentEvent("tool_result", {"name": call.name, "result": result}))
            prompt = str(result)
        except (TypeError, ValueError, KeyError, PermissionError) as exc:
            events.append(AgentEvent("error", {"error": str(exc)}))
            return "I could not safely execute that request.", events
    events.append(AgentEvent("limit", {"max_steps": max_steps}))
    return "The agent reached its step limit.", events

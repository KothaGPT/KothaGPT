"""Agent execution runner bridging ``services.agents.loop`` to the API backend.

The runner turns a configured agent into an executable loop:

1. ``decide`` asks the active chat backend to generate a response for the
   current prompt + conversation history.
2. If the response contains a tool call, the ``ToolRegistry`` executes it
   (permission-checked) and the result is fed back as the next prompt.
3. The loop repeats until the model emits no tool call or the step limit is hit.

This wires ``services/agents`` (loop + registry + permissions) into the API's
agent run endpoint so agents actually execute tools instead of returning a
canned reply.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ...agents.loop import AgentEvent
from ...agents.loop import run_agent as loop_run_agent
from ...agents.permissions import PermissionGate
from ...agents.registry import ToolRegistry
from ..api.backend import get_active_backend
from ..api.schemas import Agent, AgentRun, ChatCompletionRequest, Message


def _build_registry_and_permissions(
    agent: Agent, backend_tools: list[Any]
) -> tuple[ToolRegistry, PermissionGate]:
    """Build a :class:`ToolRegistry` and :class:`PermissionGate` for an agent.

    Only tools listed in ``agent.tools`` are allowed; all registered tools are
    exposed for discovery but gated at invoke time.
    """
    from packages.agents import ToolSpec as PkgToolSpec

    registry = ToolRegistry()
    allowed: set[str] = set()
    for tool in backend_tools:
        name = tool.function.name
        spec = PkgToolSpec(
            name=name,
            description=tool.function.description,
            parameters=tool.function.parameters,
            permission="read",
            enabled=True,
        )
        registry.register(spec, _make_handler(name))
        if agent.tools and name in agent.tools:
            allowed.add(name)
    return registry, PermissionGate(allowed_tools=allowed)


def _make_handler(name: str):
    """Return a handler that delegates to the active backend's invoke_tool."""

    def handler(**arguments: Any) -> Any:
        backend = get_active_backend("tools")
        return backend.invoke_tool(name, arguments).result

    return handler


def _decide_with_backend(
    backend: Any,
    model: str,
    messages: list[Message],
) -> ChatCompletionRequest:
    """Generate a completion via the backend."""
    request = ChatCompletionRequest(
        model=model,
        messages=[Message(role=m.role, content=m.content) for m in messages],
        stream=False,
    )
    return backend.chat(request)


def run_agent_with_loop(
    agent_id: str,
    message: str,
    agent: Agent,
    backend: Any,
    max_steps: int = 8,
) -> AgentRun:
    """Execute an agent through the decide -> tool -> result loop.

    Returns a populated :class:`AgentRun` with the full event transcript.
    """
    run_id = f"run_{uuid.uuid4().hex[:24]}"
    now = int(time.time())

    run = AgentRun(
        id=run_id,
        agent_id=agent_id,
        status="running",
        messages=[],
        created_at=now,
        updated_at=now,
    )

    backend_tools = backend.list_tools()
    registry, permissions = _build_registry_and_permissions(agent, backend_tools)

    def decide(prompt: str, events: list[AgentEvent]) -> str | None:
        # Rebuild the full message list: system + user + tool results so far.
        msgs: list[Message] = [
            Message(
                role="system",
                content=agent.instructions or "You are a helpful assistant.",
            ),
            Message(role="user", content=message),
        ]
        for event in events:
            if event.kind == "tool_result":
                payload = event.payload
                msgs.append(
                    Message(
                        role="tool",
                        content=str(payload.get("result", "")),
                        name=payload.get("name", "tool"),
                    )
                )
            elif event.kind == "error":
                msgs.append(
                    Message(
                        role="assistant",
                        content=f"[tool error] {payload.get('error', '')}",
                    )
                )

        try:
            response = backend.chat(
                ChatCompletionRequest(
                    model=agent.model,
                    messages=msgs,
                    stream=False,
                )
            )
        except Exception:  # noqa: BLE001 — degrade gracefully
            return None

        content = response.choices[0].message.content if response.choices else ""
        if not content.strip():
            return None
        return content

    try:
        final_output, events = loop_run_agent(
            message,
            decide=decide,
            registry=registry,
            permissions=permissions,
            max_steps=max_steps,
        )
    except Exception as exc:  # noqa: BLE001 — never fail a run hard
        run.status = "failed"
        run.messages = [
            {
                "role": "system",
                "content": agent.instructions or "You are a helpful assistant.",
            },
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Agent run failed: {exc}"},
        ]
        run.output = f"Agent run failed: {exc}"
        run.updated_at = int(time.time())
        return run

    # Convert events into stored messages.
    stored_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": agent.instructions or "You are a helpful assistant.",
        },
        {"role": "user", "content": message},
    ]
    assistant_content_parts: list[str] = []

    for event in events:
        if event.kind == "tool_call":
            stored_messages.append(
                {
                    "role": "assistant",
                    "content": event.payload.get("output", ""),
                    "tool_calls": [{"name": "tool", "arguments": {}}],
                }
            )
        elif event.kind == "tool_result":
            stored_messages.append(
                {
                    "role": "tool",
                    "name": event.payload.get("name", "tool"),
                    "content": str(event.payload.get("result", "")),
                }
            )
        elif event.kind == "error":
            stored_messages.append(
                {"role": "assistant", "content": f"[error] {event.payload.get('error', '')}"}
            )

    if final_output and not final_output.startswith(("I could not", "The agent reached")):
        assistant_content_parts.append(final_output)
    elif events:
        # The last tool result becomes the assistant content if no final text.
        for event in reversed(events):
            if event.kind == "tool_result":
                assistant_content_parts.append(str(event.payload.get("result", "")))
                break

    assistant_content = "\n".join(assistant_content_parts) or final_output or ""
    stored_messages.append({"role": "assistant", "content": assistant_content})

    run.messages = stored_messages
    run.output = assistant_content
    run.status = "completed"
    run.updated_at = int(time.time())
    return run


__all__ = ["run_agent_with_loop"]

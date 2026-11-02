import pytest

from services.agents.function_calling import parse_function_call
from services.agents.loop import run_agent
from services.agents.permissions import PermissionGate
from services.agents.registry import ToolRegistry, ToolSpec


def test_registry_validates_and_invokes_tools() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("add", "Add numbers", {"properties": {"a": {}, "b": {}}, "required": ["a", "b"]}),
        lambda a, b: a + b,
    )
    assert registry.invoke("add", {"a": 2, "b": 3}) == 5
    with pytest.raises(ValueError):
        registry.invoke("add", {"a": 2})


def test_agent_denies_unapproved_high_risk_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("run", "Execute", {"properties": {}}, permission="execute"), lambda: "done"
    )
    gate = PermissionGate({"run"})
    output, events = run_agent(
        "go",
        decide=lambda *_: '{"name":"run","arguments":{}}',
        registry=registry,
        permissions=gate,
        max_steps=2,
    )
    assert "safely" in output
    assert events[-1].kind == "error"


def test_function_call_parser_accepts_json_arguments() -> None:
    call = parse_function_call('{"name":"search","arguments":"{\\"query\\":\\"docs\\"}"}')
    assert call.arguments == {"query": "docs"}


def test_agent_handles_malformed_function_call_gracefully() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("run", "Execute", {"properties": {}}), lambda: "done")
    gate = PermissionGate({"run"})
    output, events = run_agent(
        "go", decide=lambda *_: '{"name":123}', registry=registry, permissions=gate, max_steps=2
    )
    assert "safely" in output
    assert events[-1].kind == "error"

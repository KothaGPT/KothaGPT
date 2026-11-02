"""Tests for packages.agents — tool registry, permission gate, agent loop."""

from packages.agents import AgentSpec, PermissionGate, ToolRegistry, ToolSpec, run_agent


def test_tool_registry_register_and_list():
    registry = ToolRegistry()
    called = []

    def handler(x: int) -> int:
        called.append(x)
        return x * 2

    registry.register(
        ToolSpec(
            name="double",
            description="doubles x",
            parameters={"properties": {"x": {"type": "integer"}}, "required": ["x"]},
        ),
        handler,
    )
    specs = registry.list()
    assert len(specs) == 1
    assert specs[0].name == "double"

    result = registry.invoke("double", {"x": 21}, allowed={"double"})
    assert result == 42
    assert called == [21]


def test_tool_registry_duplicate_raises():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="foo"), lambda: None)
    try:
        registry.register(ToolSpec(name="foo"), lambda: None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for duplicate tool")


def test_permission_gate_denies_unallowed():
    gate = PermissionGate(allowed_tools={"read_tool"})
    try:
        gate.check("write_tool", risk="low")
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError for unallowed tool")


def test_permission_gate_requires_approval_for_high_risk():
    gate = PermissionGate(allowed_tools={"exec_tool"}, approvals=set())
    try:
        gate.check("exec_tool", risk="high")
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError for unapproved high-risk tool")


def test_permission_gate_allows_approved_high_risk():
    gate = PermissionGate(allowed_tools={"exec_tool"}, approvals={"exec_tool"})
    gate.check("exec_tool", risk="high")  # should not raise


def test_agent_spec_defaults():
    spec = AgentSpec(name="test-agent")
    assert spec.name == "test-agent"
    assert spec.model == "kothagpt"
    assert spec.tools == []
    assert spec.temperature == 0.7


def test_run_agent_loop_calls_tool_and_returns():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            parameters={"properties": {"text": {"type": "string"}}, "required": ["text"]},
        ),
        lambda text: {"echoed": text},
    )

    def decide(prompt, events):
        if len(events) > 2:
            return None
        return '{"name": "echo", "arguments": {"text": "hello"}}'

    gate = PermissionGate(allowed_tools={"echo"})
    output, events = run_agent("say hi", decide=decide, registry=registry, permissions=gate)
    assert output == "{'echoed': 'hello'}"
    kinds = [e.kind for e in events]
    assert "user" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds


def test_run_agent_respects_step_limit():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="noop", parameters={"properties": {}, "required": []}),
        lambda: "done",
    )

    def decide(prompt, events):
        return '{"name": "noop", "arguments": {}}'

    gate = PermissionGate(allowed_tools={"noop"})
    _, events = run_agent(
        "loop forever", decide=decide, registry=registry, permissions=gate, max_steps=2
    )
    assert any(e.kind == "limit" for e in events)

"""Tests for packages.tools — canonical tool definitions and handlers."""

from packages.tools import (
    calculator_handler,
    calculator_spec,
    code_spec,
    current_time_handler,
    current_time_spec,
    register_default_tools,
    web_search_handler,
    web_search_spec,
)


def test_calculator_spec_and_handler():
    assert calculator_spec.name == "calculator"
    assert calculator_spec.permission == "read"
    result = calculator_handler(expression="(2 + 3) * 4")
    assert result["value"] == 20
    assert result["expression"] == "(2 + 3) * 4"


def test_calculator_rejects_unsafe():
    result = calculator_handler(expression="__import__('os')")
    assert "error" in result["value"].lower() or not result.get("success", True)


def test_current_time_spec_and_handler():
    assert current_time_spec.name == "current_time"
    result = current_time_handler()
    assert "utc" in result
    assert "T" in result["utc"]


def test_web_search_spec_and_handler():
    assert web_search_spec.name == "web_search"
    result = web_search_handler(query="bangla", max_results=3)
    assert result["query"] == "bangla"
    assert result["max_results"] == 3
    assert len(result["results"]) == 1


def test_code_spec_permission():
    assert code_spec.name == "code"
    assert code_spec.permission == "execute"


def test_default_tools_register_all():
    from packages.agents import ToolRegistry

    registry = ToolRegistry()
    register_default_tools(registry)
    names = {t.name for t in registry.list()}
    assert {"calculator", "current_time", "web_search", "code"} <= names

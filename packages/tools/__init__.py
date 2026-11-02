"""packages/tools — canonical tool definitions and invocation wrappers.

Dependency-light tool implementations consumable by ``packages.agents.ToolRegistry``
and the API tool surface. Each tool exposes a :class:`packages.agents.ToolSpec`
plus a handler callable.

Tools are intentionally sandboxed where they execute code or access the
network; the sandbox lives in ``services.security.sandbox`` when available.
"""

from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from packages.agents import ToolSpec

__all__ = [
    "calculator_handler",
    "calculator_spec",
    "code_handler",
    "code_spec",
    "current_time_handler",
    "current_time_spec",
    "default_tools",
    "web_search_handler",
    "web_search_spec",
]


# ---------------------------------------------------------------------------
# calculator — safe arithmetic expression evaluator
# ---------------------------------------------------------------------------

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expression: str) -> float | str:
    if len(expression) > 200:
        return "error: expression too long"
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
        value = _eval_node(tree.body, depth=0)
        if not math.isfinite(float(value)) or abs(value) > 1_000_000_000:
            return "error: result out of bounds"
        return value
    except (ArithmeticError, SyntaxError, ValueError, TypeError) as exc:
        return f"error: {exc}"


def _eval_node(node: ast.AST, depth: int) -> int | float:
    if depth > 20:
        raise ValueError("expression too deep")
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if abs(node.value) > 1_000_000:
            raise ValueError("number out of bounds")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent out of bounds")
        return _BINARY_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


calculator_spec = ToolSpec(
    name="calculator",
    description="Evaluate a basic arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. (2 + 3) * 4",
            }
        },
        "required": ["expression"],
    },
    permission="read",
    enabled=True,
)


def calculator_handler(expression: str) -> dict[str, Any]:
    return {"expression": expression, "value": _safe_eval(str(expression))}


# ---------------------------------------------------------------------------
# current_time — return current UTC time
# ---------------------------------------------------------------------------

current_time_spec = ToolSpec(
    name="current_time",
    description="Get the current UTC time.",
    parameters={"type": "object", "properties": {}},
    permission="read",
    enabled=True,
)


def current_time_handler() -> dict[str, str]:
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# ---------------------------------------------------------------------------
# web_search — deterministic mock search (replace with real provider later)
# ---------------------------------------------------------------------------

web_search_spec = ToolSpec(
    name="web_search",
    description="Search the web (mock).",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    permission="read",
    enabled=True,
)


def web_search_handler(query: str, max_results: int = 5) -> dict[str, Any]:
    return {
        "query": query,
        "max_results": max_results,
        "results": [{"title": "Mock result", "snippet": "Deterministic search stub."}],
    }


# ---------------------------------------------------------------------------
# code — sandboxed Python execution (wraps services.agents.tools.code when present)
# ---------------------------------------------------------------------------

try:  # pragma: no cover — optional dependency on services layer
    from services.agents.tools import run_code as _services_run_code  # type: ignore

    def code_handler(
        code: str, language: str = "python", timeout_sec: int | None = None
    ) -> dict[str, Any]:
        result = _services_run_code(code, language=language, timeout_sec=timeout_sec or 10)
        return result.to_dict()

except Exception:  # noqa: BLE001 — services layer unavailable in minimal env

    def code_handler(
        code: str, language: str = "python", timeout_sec: int | None = None
    ) -> dict[str, Any]:
        return {
            "success": False,
            "language": language,
            "error": "services.agents.tools not available in this environment",
        }


code_spec = ToolSpec(
    name="code",
    description=(
        "Execute Python code in a sandbox with no network, strict CPU/memory "
        "budgets, and stdout/stderr capture. Use for calculations, data "
        "transformations, and verifying code behavior."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "language": {
                "type": "string",
                "description": "Language alias; currently only python is supported.",
                "default": "python",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Per-call execution timeout in seconds.",
                "default": 10,
            },
        },
        "required": ["code"],
    },
    permission="execute",
    enabled=True,
)


def default_tools() -> list[tuple[ToolSpec, Any]]:
    """Return the canonical set of (spec, handler) pairs.

    Suitable for registering into a fresh :class:`packages.agents.ToolRegistry`.
    """
    return [
        (calculator_spec, calculator_handler),
        (current_time_spec, current_time_handler),
        (web_search_spec, web_search_handler),
        (code_spec, code_handler),
    ]


def register_default_tools(registry) -> None:
    """Register every canonical tool into ``registry``."""
    for spec, handler in default_tools():
        registry.register(spec, handler)

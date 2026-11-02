"""Agent tool implementations (WS-3..WS-7).

Tools live under ``services/agents/tools/`` and are gated by the permission
system (``services/agents/permissions.py``) and sandbox
(``services/security/sandbox.py``). Each tool exposes a ``ToolSpec``-shaped
handler consumable by ``services/agents/registry.py``.
"""

from .code import CodeExecutionTool, CodeResult, default_handler, default_tool_spec, run_code

__all__ = [
    "CodeExecutionTool",
    "CodeResult",
    "default_handler",
    "default_tool_spec",
    "run_code",
]
"""Code execution tool (WS-5)."""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from ...security.sandbox import Sandbox

__all__ = ["CodeExecutionTool", "run_code", "CodeResult"]

DEFAULT_CPU_SEC = 5
DEFAULT_MEM_MB = 256
DEFAULT_TIMEOUT_SEC = 10

_SUPPORTED_LANGUAGES = {"python", "py", "python3"}


@dataclass
class CodeResult:
    """Outcome of a code execution attempt."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    language: str = "python"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "language": self.language,
            "error": self.error,
        }


def _normalize_language(language: str) -> str:
    lang = (language or "python").strip().lower()
    if lang in {"python", "py", "python3"}:
        return "python"
    return lang


def _syntax_valid(code: str, language: str) -> tuple[bool, str | None]:
    """Return (valid, error_message)."""
    if language != "python":
        return True, None
    try:
        ast.parse(code, mode="exec")
        return True, None
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


def run_code(
    code: str,
    *,
    language: str = "python",
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    cpu_sec: int = DEFAULT_CPU_SEC,
    mem_mb: int = DEFAULT_MEM_MB,
    allow_network: bool = False,
) -> CodeResult:
    """Execute code in a sandboxed subprocess.

    Network egress is disabled by default; the sandbox enforces CPU/memory
    limits and confines file access to a scratch directory.
    """
    normalized = _normalize_language(language)
    if normalized not in _SUPPORTED_LANGUAGES:
        return CodeResult(
            success=False,
            language=normalized,
            error=f"unsupported language: {language}",
        )
    valid, syntax_error = _syntax_valid(code, normalized)
    if not valid:
        return CodeResult(
            success=False,
            language=normalized,
            error=syntax_error,
        )

    sandbox = Sandbox(cpu_sec=cpu_sec, mem_mb=mem_mb, allow_network=allow_network)
    cmd = [sys.executable, "-c", code]
    try:
        completed = sandbox.run(cmd, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return CodeResult(
            success=False,
            language=normalized,
            timed_out=True,
            error=f"execution timed out after {timeout_sec}s",
        )
    except PermissionError as exc:
        return CodeResult(
            success=False,
            language=normalized,
            error=str(exc),
        )

    success = completed.returncode == 0
    return CodeResult(
        success=success,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        language=normalized,
    )


@dataclass
class CodeExecutionTool:
    """ToolSpec-shaped handler for sandboxed code execution."""

    cpu_sec: int = DEFAULT_CPU_SEC
    mem_mb: int = DEFAULT_MEM_MB
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    allow_network: bool = False

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandbox with no network, strict CPU/memory "
            "budgets, and stdout/stderr capture. Use for calculations, data "
            "transformations, and verifying code behavior."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
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
                    "default": DEFAULT_TIMEOUT_SEC,
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str, language: str = "python", timeout_sec: int | None = None) -> CodeResult:
        return run_code(
            code,
            language=language,
            timeout_sec=timeout_sec or self.timeout_sec,
            cpu_sec=self.cpu_sec,
            mem_mb=self.mem_mb,
            allow_network=self.allow_network,
        )

    def __call__(self, **arguments: Any) -> dict[str, Any]:
        code = arguments.get("code")
        language = arguments.get("language", "python")
        timeout_sec = arguments.get("timeout_sec")
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        if not isinstance(language, str):
            raise ValueError("language must be a string")
        if timeout_sec is not None and (
            not isinstance(timeout_sec, int) or isinstance(timeout_sec, bool) or timeout_sec < 1
        ):
            raise ValueError("timeout_sec must be a positive integer")
        result = self.execute(code, language=language, timeout_sec=timeout_sec)
        return result.to_dict()


def default_tool_spec() -> dict[str, Any]:
    """Return the ToolSpec dict for registering code execution in the registry."""
    tool = CodeExecutionTool()
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "permission": "execute",
        "enabled": True,
    }


def default_handler() -> tuple[dict[str, Any], CodeExecutionTool]:
    return default_tool_spec(), CodeExecutionTool()
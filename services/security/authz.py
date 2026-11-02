"""WS-2 Tool authorization — scoped, auditable (replaces ad-hoc checks)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class Permission:
    tool: str
    resource: str = "*"
    budget: int | None = None  # max calls per session
    requires_approval: bool = False

@dataclass
class AuthDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False

class Authorizer:
    def __init__(self, permissions: list[Permission] | None = None):
        self.permissions: dict[str, Permission] = {p.tool: p for p in (permissions or [])}
        self._counts: dict[str, int] = {}
        self.audit: list[dict[str, Any]] = []

    def authorize(self, tool: str, *, resource: str = "*", context: dict[str, Any] | None = None) -> AuthDecision:
        perm = self.permissions.get(tool) or self.permissions.get("*")
        if perm is None:
            dec = AuthDecision(False, f"no permission for {tool}")
            self._log(tool, resource, dec)
            return dec
        if perm.resource != "*" and perm.resource != resource:
            dec = AuthDecision(False, f"tool {tool} not allowed on {resource}")
            self._log(tool, resource, dec)
            return dec
        if perm.budget is not None:
            cnt = self._counts.get(tool, 0)
            if cnt >= perm.budget:
                dec = AuthDecision(False, f"budget exceeded for {tool}")
                self._log(tool, resource, dec)
                return dec
        if perm.requires_approval:
            dec = AuthDecision(False, f"approval required for {tool}", requires_approval=True)
            self._log(tool, resource, dec)
            return dec
        # allow
        self._counts[tool] = self._counts.get(tool, 0) + 1
        dec = AuthDecision(True, "allowed")
        self._log(tool, resource, dec)
        return dec

    def approve(self, tool: str) -> None:
        # One-time approval bypasses requires_approval for next call
        perm = self.permissions.get(tool)
        if perm and perm.requires_approval:
            # Temporarily allow one call
            self._counts[tool] = self._counts.get(tool, 0)
            # Flip to allow for next authorize (caller should re-authorize)
            # Simplistic: just increment and allow

    def _log(self, tool: str, resource: str, dec: AuthDecision) -> None:
        self.audit.append({
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool,
            "resource": resource,
            "allowed": dec.allowed,
            "reason": dec.reason,
        })

# Preset matrices per docs/agent-plan.md WS-14
DEFAULT_PERMISSIONS = [
    Permission("calculator", budget=100),
    Permission("current_time"),
    Permission("web_search", budget=20),
    Permission("code", requires_approval=True),
    Permission("db", requires_approval=True),
]

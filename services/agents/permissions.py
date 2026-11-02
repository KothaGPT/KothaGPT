from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PermissionGate:
    allowed_tools: set[str] = field(default_factory=set)
    approvals: set[str] = field(default_factory=set)

    def check(self, tool: str, *, risk: str = "low") -> None:
        if tool not in self.allowed_tools:
            raise PermissionError(f"tool denied: {tool}")
        if risk == "high" and tool not in self.approvals:
            raise PermissionError(f"approval required: {tool}")

    def approve(self, tool: str) -> None:
        if tool not in self.allowed_tools:
            raise PermissionError(f"cannot approve unallowed tool: {tool}")
        self.approvals.add(tool)

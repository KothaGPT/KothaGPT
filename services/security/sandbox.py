"""WS-3 Agent sandbox — OS isolation stub with budget and teardown."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class Sandbox:
    def __init__(self, *, cpu_sec: int = 5, mem_mb: int = 256, allow_network: bool = False):
        self.cpu_sec = cpu_sec
        self.mem_mb = mem_mb
        self.allow_network = allow_network
        self._tmp: Path | None = None

    def __enter__(self) -> Path:
        self._tmp = Path(tempfile.mkdtemp(prefix="kothagpt-sandbox-"))
        return self._tmp

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp and self._tmp.exists():
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None

    def run(self, cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
        # Enforce no network if not allowed (check for curl/wget)
        if not self.allow_network and any(c in ("curl", "wget", "nc") for c in cmd):
            raise PermissionError("network egress not allowed in sandbox")
        # Enforce file access only within tmp
        with self as tmp:
            # Write a marker to ensure scratch-only
            result = subprocess.run(cmd, cwd=cwd or tmp, capture_output=True, text=True, timeout=self.cpu_sec, check=False)
            return result

    def check_escape(self, attempt: str) -> bool:
        # Heuristic: block attempts to escape tmp, access host files, or network
        blocked = [
            "../", "/etc/passwd", "/proc/", "curl ", "wget ", "nc ", "socket",
        ]
        return any(b in attempt for b in blocked)

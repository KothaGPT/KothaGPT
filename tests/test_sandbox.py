"""WS-3: Agent sandbox — budget, teardown, network and path-escape guards."""

from __future__ import annotations

import subprocess
import sys

import pytest

from services.security.sandbox import Sandbox


def test_sandbox_context_manager_creates_and_tears_down_tmp():
    with Sandbox() as tmp:
        assert tmp.is_dir()
        marker = tmp / "marker.txt"
        marker.write_text("scratch-only")
        assert marker.read_text() == "scratch-only"
    assert not tmp.exists()


def test_sandbox_run_executes_command():
    with Sandbox(cpu_sec=5) as tmp:
        result = Sandbox(cpu_sec=5).run([sys.executable, "-c", "print(1 + 1)"], cwd=tmp)
    assert result.returncode == 0
    assert result.stdout.strip() == "2"


def test_sandbox_run_respects_cpu_budget():
    with Sandbox(cpu_sec=1) as tmp, pytest.raises(subprocess.TimeoutExpired):
        Sandbox(cpu_sec=1).run(
            [sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp
        )


def test_sandbox_blocks_network_egress():
    for cmd in (["curl", "http://evil.example"], ["wget", "http://evil.example"], ["nc", "-l", "1234"]):
        with Sandbox(cpu_sec=5, allow_network=False) as tmp, pytest.raises(PermissionError):
            Sandbox(cpu_sec=5, allow_network=False).run(cmd, cwd=tmp)


def test_sandbox_check_escape_blocks_traversal():
    s = Sandbox()
    assert s.check_escape("../etc/passwd") is True
    assert s.check_escape("/etc/passwd") is True
    assert s.check_escape("/proc/self/cmdline") is True
    assert s.check_escape("curl http://evil.example") is True
    assert s.check_escape("wget http://evil.example") is True
    assert s.check_escape("nc -l 1234") is True
    assert s.check_escape("read a normal file") is False


def test_sandbox_allows_network_when_enabled():
    with Sandbox(cpu_sec=5, allow_network=True) as tmp:
        result = Sandbox(cpu_sec=5, allow_network=True).run(
            ["echo", "allowed"], cwd=tmp
        )
    assert result.returncode == 0


def test_sandbox_defaults():
    s = Sandbox()
    assert s.cpu_sec == 5
    assert s.mem_mb == 256
    assert s.allow_network is False
    assert s._tmp is None

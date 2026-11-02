"""Tests for the platform-aware bootstrap dispatcher (docs/bootstrap-migration-plan.md)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap.sh"

requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def run_bootstrap(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BOOTSTRAP), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def preview(platform: str) -> str:
    result = run_bootstrap("--dry-run", "--platform", platform)
    assert result.returncode == 0, result.stderr
    return result.stdout


def venv_is_ready(platform: str) -> bool:
    leaf = "Scripts/python.exe" if platform == "windows" else "bin/python"
    return (REPO_ROOT / ".venv" / leaf).exists()


def assert_install_line(out: str, tool: str, fragment: str) -> None:
    if shutil.which(tool):
        assert fragment not in out, f"{tool} present but install step emitted"
    else:
        assert fragment in out, f"{tool} missing but no install step emitted"


@requires_bash
@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_dry_run_unix_preview(platform: str) -> None:
    out = preview(platform)
    assert f"bootstrap preview — {platform}" in out
    assert ".venv/bin/activate" in out
    assert "services/api/requirements.txt" in out
    assert "corepack enable" in out
    assert "pnpm install" in out
    if venv_is_ready(platform):
        assert "-m venv" not in out
    else:
        assert "python3 -m venv" in out


@requires_bash
def test_dry_run_installs_missing_toolchains() -> None:
    go_fragments = {
        "linux": "golang",
        "macos": "brew install go",
        "windows": "GoLang.Go",
    }
    for platform, go_fragment in go_fragments.items():
        out = preview(platform)
        assert_install_line(out, "go", go_fragment)
        if shutil.which("cargo"):
            assert "install_rust" not in out
        else:
            assert "install_rust" in out


@requires_bash
def test_linux_uses_detected_package_manager() -> None:
    out = preview("linux")
    signatures = {"apt-get": "apt-get install", "dnf": "dnf install", "pacman": "pacman -S"}
    available = {pm: sig for pm, sig in signatures.items() if shutil.which(pm)}
    needs_install = shutil.which("python3") is None or shutil.which("go") is None
    if available and needs_install:
        assert any(sig in out for sig in available.values())
    elif not available:
        assert "sudo" not in out


@requires_bash
def test_dry_run_windows_preview() -> None:
    out = preview("windows")
    assert "bootstrap preview — windows" in out
    assert ".venv/Scripts/activate" in out
    if venv_is_ready("windows"):
        assert "-m venv" not in out
    else:
        assert "py -m venv" in out


@requires_bash
def test_preview_steps_are_sequentially_numbered() -> None:
    out = preview("linux")
    nums = [int(m.group(1)) for m in re.finditer(r"^ +(\d+)\. ", out, re.MULTILINE)]
    assert nums == list(range(1, len(nums) + 1))
    assert len(nums) >= 4


@requires_bash
def test_preview_alias_matches_dry_run() -> None:
    assert run_bootstrap("--preview", "--platform", "linux").stdout == preview("linux")


@requires_bash
def test_explicit_platform_overrides_detection() -> None:
    out = preview("windows")
    assert ".venv/Scripts/activate" in out
    assert ".venv/bin/activate" not in out


@requires_bash
def test_unknown_platform_fails() -> None:
    result = run_bootstrap("--dry-run", "--platform", "solaris")
    assert result.returncode != 0
    assert "invalid --platform" in result.stderr


@requires_bash
def test_unknown_option_fails_with_usage_hint() -> None:
    result = run_bootstrap("--bad-flag")
    assert result.returncode != 0
    assert "unknown option" in result.stderr


@requires_bash
def test_help_exits_zero() -> None:
    result = run_bootstrap("--help")
    assert result.returncode == 0
    assert "--platform" in result.stdout


@requires_bash
def test_dry_run_leaves_repo_tree_unchanged() -> None:
    before = sorted(p.name for p in REPO_ROOT.iterdir())
    preview("linux")
    preview("windows")
    after = sorted(p.name for p in REPO_ROOT.iterdir())
    assert before == after


@requires_bash
def test_non_interactive_execution_requires_yes() -> None:
    result = run_bootstrap("--platform", "linux")
    assert result.returncode != 0
    assert "--yes" in result.stderr


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_platform_modules_define_contract(platform: str) -> None:
    text = (REPO_ROOT / "scripts" / platform / "bootstrap.sh").read_text(encoding="utf-8")
    assert "bootstrap_check()" in text
    assert "bootstrap_steps()" in text


def test_windows_ps1_mirrors_shim() -> None:
    ps1 = (REPO_ROOT / "scripts" / "windows" / "bootstrap.ps1").read_text(encoding="utf-8")
    assert "Activate.ps1" in ps1
    assert "services/api/requirements.txt" in ps1
    assert "DryRun" in ps1
    for winget_id in ("Python.Python.3.12", "GoLang.Go", "Rustlang.Rustup"):
        assert winget_id in ps1


def test_makefile_bootstrap_delegates_to_script() -> None:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^bootstrap:\n\t(.*)$", text, re.MULTILINE)
    assert match, "bootstrap target missing from Makefile"
    assert "scripts/bootstrap.sh --yes" in match.group(1)

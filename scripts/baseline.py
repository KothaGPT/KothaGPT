"""Repository baseline validator (KGPT-001).

Verifies that the repository is in a supported, reproducible state:

1. Python version meets the declared floor (>=3.11).
2. Required top-level directories and manifests exist.
3. The data pipeline has a CURRENT version (or is initialized).
4. The tokenizer artifact exists (or is flagged as missing).
5. The API requirements file is present.
6. The model registry exposes at least the canonical families.
7. The evals suites directory is non-empty.
8. The Makefile exposes the canonical targets.

Exits 0 when every mandatory check passes; otherwise exits 1 with a
concise list of failures. Warnings are reported but do not fail the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _report(label: str, passed: bool, detail: str, *, warning: bool = False) -> tuple[bool, bool, str]:
    status = "WARN" if warning else ("OK" if passed else "FAIL")
    print(f"[{status}] {label}: {detail}")
    return passed, warning, detail


def _required_paths() -> list[str]:
    return [
        "data",
        "ml",
        "services",
        "tests",
        "configs",
        "docs",
        "README.md",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ]


def _check_python_version() -> tuple[bool, bool, str]:
    ok = sys.version_info >= (3, 11)
    return _report(
        "Python version",
        ok,
        f"{sys.version.split()[0]} (requires >=3.11)",
    )


def _check_repo_paths() -> tuple[bool, bool, str]:
    required = _required_paths()
    missing = [name for name in required if not (REPO_ROOT / name).exists()]
    return _report(
        "Required repo paths",
        not missing,
        ", ".join(required) if not missing else f"missing: {', '.join(missing)}",
    )


def _check_dataset_version() -> tuple[bool, bool, str]:
    current = REPO_ROOT / "data" / "processed" / "CURRENT"
    if current.exists():
        version = current.read_text(encoding="utf-8").strip()
        return _report("Dataset version", True, version or "present", warning=False)
    return _report(
        "Dataset version",
        True,
        "not initialized; run 'make data' or 'python -m data.pipeline.cli run'",
        warning=True,
    )


def _check_tokenizer_artifact() -> tuple[bool, bool, str]:
    tokenizer = REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "best" / "tokenizer.json"
    exists = tokenizer.exists()
    return _report(
        "Tokenizer artifact",
        True,
        "present" if exists else "missing; run 'make tokenizer-build'",
        warning=not exists,
    )


def _check_api_requirements() -> tuple[bool, bool, str]:
    path = REPO_ROOT / "services" / "api" / "requirements.txt"
    return _report("API requirements", path.exists(), str(path), warning=not path.exists())


def _check_makefile_targets() -> tuple[bool, bool, str]:
    makefile = REPO_ROOT / "Makefile"
    if not makefile.exists():
        return _report("Makefile", False, "missing")
    text = makefile.read_text(encoding="utf-8")
    required_targets = [
        "bootstrap",
        "test",
        "lint",
        "data-validate",
        "tokenizer-build",
        "serve-proto",
        "sft-smoke",
        "eval-bangla",
        "plans-check",
        "baseline",
    ]
    missing = [t for t in required_targets if f"{t}:" not in text]
    return _report(
        "Makefile targets",
        not missing,
        ", ".join(required_targets) if not missing else f"missing: {', '.join(missing)}",
    )


def _check_evals_suites() -> tuple[bool, bool, str]:
    suites_dir = REPO_ROOT / "evals" / "suites"
    if not suites_dir.is_dir():
        return _report("Eval suites", False, "evals/suites/ missing")
    suites = sorted(p.name for p in suites_dir.glob("*.yaml"))
    return _report(
        "Eval suites",
        len(suites) >= 4,
        f"{len(suites)} suites: {', '.join(suites)}",
    )


def _check_model_registry() -> tuple[bool, bool, str]:
    try:
        from ml.inference.registry import get_registry

        records = get_registry().list_models()
        ids = {r.id for r in records}
        required = {"kothagpt", "kothagpt-small", "kothagpt-embed", "kothagpt-rerank"}
        missing = required - ids
        return _report(
            "Model registry",
            not missing,
            f"{len(records)} records; missing: {', '.join(sorted(missing))}" if missing else f"{len(records)} canonical families present",
        )
    except Exception as exc:  # noqa: BLE001
        return _report("Model registry", False, f"unable to load registry: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Validate the local repository baseline and print a concise status report."""
    del argv
    print("KothaGPT baseline check")
    print(f"repo_root={REPO_ROOT}")

    checks = [
        _check_python_version(),
        _check_repo_paths(),
        _check_dataset_version(),
        _check_tokenizer_artifact(),
        _check_api_requirements(),
        _check_makefile_targets(),
        _check_evals_suites(),
        _check_model_registry(),
    ]

    failed = [detail for passed, warning, detail in checks if not passed and not warning]
    if failed:
        print("\nRepository baseline check: FAILED")
        return 1

    print("\nRepository baseline check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
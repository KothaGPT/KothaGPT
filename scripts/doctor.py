from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _report(label: str, passed: bool, detail: str, *, warning: bool = False) -> tuple[bool, bool, str]:
    status = "WARN" if warning else ("OK" if passed else "FAIL")
    print(f"[{status}] {label}: {detail}")
    return passed, warning, detail


def main(argv: list[str] | None = None) -> int:
    """Validate the local repository baseline and print a concise status report."""
    del argv
    print("KothaGPT doctor")
    print(f"repo_root={REPO_ROOT}")

    checks: list[tuple[bool, bool, str]] = []

    version_ok = sys.version_info >= (3, 11)
    checks.append(_report("Python version", version_ok, f"{sys.version.split()[0]} (requires >=3.11)"))

    required_paths = [
        "data",
        "ml",
        "services",
        "tests",
        "configs",
        "docs",
        "README.md",
    ]
    missing = [name for name in required_paths if not (REPO_ROOT / name).exists()]
    checks.append(_report("Required repo paths", not missing, ", ".join(required_paths) if not missing else f"missing: {', '.join(missing)}"))

    processed_current = REPO_ROOT / "data" / "processed" / "CURRENT"
    if processed_current.exists():
        checks.append(_report("Dataset version", True, processed_current.read_text(encoding="utf-8").strip() or "present", warning=False))
    else:
        checks.append(_report("Dataset version", True, "not initialized; run 'make data' or 'python -m data.pipeline.cli run'", warning=True))

    tokenizer_best = REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "best" / "tokenizer.json"
    tokenizer_exists = tokenizer_best.exists()
    checks.append(_report("Tokenizer artifact", True, "present" if tokenizer_exists else "missing; run 'make tokenizer-build'", warning=not tokenizer_exists))

    api_requirements = REPO_ROOT / "services" / "api" / "requirements.txt"
    checks.append(_report("API requirements", api_requirements.exists(), str(api_requirements), warning=not api_requirements.exists()))

    failed = [detail for passed, warning, detail in checks if not passed and not warning]
    if failed:
        print("\nRepository baseline check: FAILED")
        return 1

    print("\nRepository baseline check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

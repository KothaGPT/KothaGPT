"""Run the structured red-team drill suite with the shared adversarial runner."""

from __future__ import annotations

import sys

from . import adversarial


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(argument == "--suite" or argument.startswith("--suite=") for argument in arguments):
        arguments = ["--suite", str(adversarial._SUITE_DIR / "redteam.yaml"), *arguments]
    return adversarial.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

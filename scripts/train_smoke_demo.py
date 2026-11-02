"""Minimal end-to-end training pipeline demo.

Demonstrates the full flow:

    data/raw → data/processed → tokenizer → model checkpoint → eval

Run the parts that don't require torch (data pipeline + tokenizer) directly:

    python scripts/train_smoke_demo.py --stage data --stage tokenizer

Run training + eval only when torch is installed:

    python scripts/train_smoke_demo.py --stage train --stage eval

Without torch, train/eval stages print instructions and exit cleanly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    print(f"[demo] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=_REPO_ROOT, check=check, **kwargs)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def stage_data() -> None:
    """Build a tiny processed dataset from fixtures."""
    raw = _REPO_ROOT / "tests" / "fixtures" / "raw"
    out = _REPO_ROOT / "data" / "processed" / "demo"
    out.mkdir(parents=True, exist_ok=True)

    # Use the data pipeline CLI to normalize + write a processed corpus.
    _run(
        [
            sys.executable,
            "-m",
            "data.pipeline.cli",
            "run",
            "--raw-dir",
            str(raw),
            "--out-root",
            str(_REPO_ROOT / "data" / "processed"),
            "--version-label",
            "demo",
            "--min-chars",
            "1",
            "--max-chars",
            "1000000",
            "--min-words",
            "1",
            "--no-gzip",
            "--dev-bypass",
        ]
    )

    # Point CURRENT at the demo dataset so downstream stages find it.
    current = _REPO_ROOT / "data" / "processed" / "CURRENT"
    current.write_text("demo\n", encoding="utf-8")
    print(f"[demo] data stage complete → {out}")


def stage_tokenizer() -> None:
    """Train a BPE tokenizer on the processed demo corpus."""
    corpus = _REPO_ROOT / "data" / "processed" / "CURRENT" / "train"
    if not corpus.is_dir():
        print("[demo] no processed corpus found; run --stage data first")
        return

    out = _REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "demo"
    out.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "ml.tokenizer.cli",
            "train",
            "--corpus",
            str(corpus),
            "--algorithm",
            "bpe",
            "--vocab-size",
            "2000",
            "--min-frequency",
            "1",
            "--out",
            str(out),
        ]
    )
    print(f"[demo] tokenizer stage complete → {out}")


def stage_train() -> None:
    """Run a short smoke training pass (requires torch)."""
    if not _have_torch():
        print(
            "[demo] torch is not installed; skipping training.\n"
            "    Install: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        )
        return

    tok = _REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "demo" / "tokenizer.json"
    if not tok.is_file():
        tok = _REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "best" / "tokenizer.json"

    _run(
        [
            sys.executable,
            "-m",
            "ml.trainer.cli",
            "run",
            "--config",
            "ml/configs/smoke.yaml",
            "--device",
            "cpu",
            "--out",
            "ml/pretrain/artifacts/demo",
            "--max-steps",
            "5",
        ]
    )
    print("[demo] train stage complete")


def stage_eval() -> None:
    """Run a quick eval smoke (requires torch for the engine backend)."""
    if not _have_torch():
        print(
            "[demo] torch is not installed; skipping eval.\n"
            "    Install: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        )
        return

    # Evaluate against the mock backend (no model weights needed).
    _run(
        [
            sys.executable,
            "-m",
            "evals.run",
            "--suite",
            "evals/suites/bangla.yaml",
            "--split",
            "all",
            "--target",
            "mock",
            "--out-dir",
            "evals/results/demo",
            "--name",
            "demo-bangla",
        ]
    )
    print("[demo] eval stage complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal end-to-end training demo")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["data", "tokenizer", "train", "eval"],
        help="Pipeline stage to run (repeatable)",
    )
    args = parser.parse_args()

    stages = args.stage or ["data", "tokenizer"]
    os.chdir(_REPO_ROOT)

    if "data" in stages:
        stage_data()
    if "tokenizer" in stages:
        stage_tokenizer()
    if "train" in stages:
        stage_train()
    if "eval" in stages:
        stage_eval()

    print("[demo] pipeline stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

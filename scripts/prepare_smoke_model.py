"""Prepare a smoke-model checkpoint for real engine-backend inference.

This script initializes a tiny KothaGPT model from ``ml/configs/smoke.yaml``,
optionally runs a few training steps over the fixture corpus, and writes a
checkpoint that the :class:`services.api.core.engine_backend.EngineBackend`
can load via ``KOTHAGPT_MODEL_CHECKPOINT`` / ``KOTHAGPT_TOKENIZER_PATH``.

Run only when torch is installed (``make gpu-env`` or CPU wheel):

    python scripts/prepare_smoke_model.py \
        --config ml/configs/smoke.yaml \
        --out ml/pretrain/artifacts/kothagpt \
        --tokenizer ml/tokenizer/artifacts/best/tokenizer.json

Without torch this script prints the steps it would take and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(config_path: Path):
    sys.path.insert(0, str(_REPO_ROOT))
    from ml.models.config import load_config

    return load_config(config_path)


def _try_torch():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def prepare(config_path: Path, out_dir: Path, tokenizer_path: Path, steps: int = 0):
    """Prepare a checkpoint directory. Returns True if a real model was saved."""
    if not _try_torch():
        print(
            "[prepare-smoke-model] torch is not installed; cannot create a real checkpoint.\n"
            "    Install torch (CPU wheel) and re-run:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        )
        return False

    import torch

    from ml.models import KothaGPT

    cfg = _load_config(config_path)
    model = KothaGPT(cfg.model)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "step": 0, "config": cfg.to_dict()},
        ckpt_dir / "best.pt",
    )

    # Mirror the tokenizer into the artifact dir so the engine can find it.
    tok_target = out_dir / "tokenizer.json"
    if tokenizer_path.is_file() and not tok_target.exists():
        tok_target.write_text(tokenizer_path.read_text(encoding="utf-8"), encoding="utf-8")

    if steps > 0:
        print(
            f"[prepare-smoke-model] note: {steps} training steps requested but "
            "this script only initializes the model. Use the trainer loop for "
            "actual training."
        )

    print(f"[prepare-smoke-model] checkpoint written to {out_dir / 'checkpoints' / 'best.pt'}")
    print(f"[prepare-smoke-model] config written to {out_dir / 'config.json'}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a smoke-model checkpoint")
    parser.add_argument("--config", default="ml/configs/smoke.yaml", help="Model config YAML")
    parser.add_argument(
        "--out",
        default="ml/pretrain/artifacts/kothagpt",
        help="Output artifact directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="ml/tokenizer/artifacts/best/tokenizer.json",
        help="Tokenizer JSON path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Training steps (informational only; initialization only)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    prepare(
        config_path=Path(args.config),
        out_dir=Path(args.out),
        tokenizer_path=Path(args.tokenizer),
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

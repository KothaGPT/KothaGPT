"""SFT CLI: python -m ml.sft.cli run --config ml/configs/sft.yaml"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ml.instruction.dataset import load_jsonl
from ml.models import KothaGPT, load_config
from ml.sft.trainer import run_sft_trainer
from ml.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--train", required=True, help="instruction jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", help="base checkpoint (optional)")
    parser.add_argument("--out", default="ml/sft/artifacts/run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--template", default="default")
    args = parser.parse_args()

    config = load_config(args.config)
    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    tokenizer = load_tokenizer(tok_path)
    records = load_jsonl(args.train)
    if config.model.vocab_size != len(tokenizer.vocab):
        raise ValueError(
            f"model vocab_size={config.model.vocab_size} does not match "
            f"tokenizer vocab_size={len(tokenizer.vocab)}"
        )
    config.data.train = str(Path(args.train).resolve())
    config.data.tokenizer_path = str(tok_path.resolve())
    model = KothaGPT(config.model)
    base_path: Path | None = None
    if args.base:
        base_path = Path(args.base)
        if not base_path.exists():
            raise SystemExit(f"base checkpoint not found: {args.base}")
        import torch

        try:
            state = torch.load(base_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise SystemExit(f"failed to load base checkpoint {args.base}: {e}") from e
        # Trainer saves {"model_state": ...}, not "model"
        if isinstance(state, dict) and "model_state" in state:
            model_state = state["model_state"]
        elif isinstance(state, dict) and all(isinstance(k, str) for k in state):
            # Assume direct state dict (weights-only)
            model_state = state
        else:
            raise SystemExit(f"base checkpoint {args.base} has no model_state")
        try:
            model.load_state_dict(model_state, strict=True)
        except Exception as e:
            raise SystemExit(f"incompatible base checkpoint {args.base}: {e}") from e

    out_dir = Path(args.out)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "best.pt"
    if base_path is not None and checkpoint_path.resolve() == base_path.resolve():
        raise ValueError("SFT output would overwrite the base checkpoint; choose another --out")

    metrics = run_sft_trainer(
        model,
        records,
        tokenizer,
        device=args.device,
        max_steps=args.max_steps if args.max_steps is not None else config.training.max_steps,
        batch_size=args.batch_size if args.batch_size is not None else config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_length=config.model.max_position_embeddings,
        template=args.template,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    try:
        import torch

        torch.save(
            {
                "model_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
            },
            temporary_checkpoint,
        )
        os.replace(temporary_checkpoint, checkpoint_path)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    metadata = {
        "base_checkpoint": str(base_path.resolve()) if base_path else None,
        "base_checkpoint_sha256": digest(base_path) if base_path else None,
        "train_data": str(Path(args.train).resolve()),
        "train_data_sha256": digest(Path(args.train)),
        "tokenizer": str(tok_path.resolve()),
        "tokenizer_sha256": digest(tok_path),
        "template": args.template,
        **metrics,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()

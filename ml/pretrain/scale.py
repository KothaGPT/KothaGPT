"""Scale training smoke — long-context + larger model via ml/trainer (WS-12)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.models import KothaGPT, load_config
from ml.tokenizer import load_tokenizer


def run_scale_smoke(config_path: str, tokenizer_path: str, max_steps: int = 5) -> dict:
    cfg = load_config(config_path)
    tok_path = Path(tokenizer_path)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    # Use tiny model for smoke, but respect max_position_embeddings for long-context
    model = KothaGPT(cfg.model)
    # Simulate scale training via short loop
    from ml.instruction.dataset import load_jsonl
    # Use instruction data as proxy for scale corpus
    recs = load_jsonl("tests/fixtures/instruction.jsonl")
    from ml.sft.trainer import run_sft_trainer
    metrics = run_sft_trainer(model, recs[:2], tok, max_steps=max_steps, max_length=cfg.model.max_position_embeddings)
    return {"scale_smoke": True, "config": str(config_path), **metrics}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ml/configs/long.yaml")
    p.add_argument("--tokenizer", default="ml/tokenizer/artifacts/best")
    p.add_argument("--max-steps", type=int, default=5)
    args = p.parse_args()
    print(run_scale_smoke(args.config, args.tokenizer, args.max_steps))

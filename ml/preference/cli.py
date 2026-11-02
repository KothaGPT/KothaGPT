"""Preference CLI: python -m ml.preference.cli --train data/processed/.../preference/train.jsonl"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.models import KothaGPT, load_config
from ml.preference.dataset import load_preference_jsonl
from ml.preference.trainer import run_dpo
from ml.tokenizer import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="DPO training")
    p.add_argument("--train", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="ml/preference/artifacts/run")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    recs = load_preference_jsonl(args.train)
    model = KothaGPT(cfg.model)
    metrics = run_dpo(model, recs, tok, device=args.device, max_steps=args.max_steps, batch_size=args.batch_size, max_length=cfg.model.max_position_embeddings)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    print(metrics)

if __name__ == "__main__":
    main()

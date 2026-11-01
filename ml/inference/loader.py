"""WS-8 Model loading system — registry-aware, quant-aware, hot-reload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.inference.registry import get_registry


def load_model(model_id: str = "kothagpt", device: str = "cpu", quant: str = "none"):
    reg = get_registry()
    rec = reg.get(model_id)
    if not rec:
        raise ValueError(f"model {model_id!r} not in registry")
    # Resolve artifact path
    artifact = Path(rec.artifact_path) if rec.artifact_path else Path(f"ml/pretrain/artifacts/{model_id}")
    # For stub, fall back to small config
    from ml.models import KothaGPT, load_config
    from ml.tokenizer import load_tokenizer

    # Determine config: prefer artifact's config.json, else small
    cfg_path = artifact / "config.json" if artifact.is_dir() else Path("ml/configs/small.yaml")
    if cfg_path.is_file():
        try:
            cfg = load_config(cfg_path)
        except Exception:
            from ml.models.config import ModelConfig
            cfg = type("Cfg", (), {"model": ModelConfig(vocab_size=698, hidden_size=128, num_layers=2, num_heads=4, max_position_embeddings=rec.context_window)})()
    else:
        from ml.models.config import ModelConfig
        cfg = type("Cfg", (), {"model": ModelConfig(vocab_size=698, hidden_size=128, num_layers=2, num_heads=4, max_position_embeddings=rec.context_window)})()

    # Tokenizer
    tok_path = Path("ml/tokenizer/artifacts/best/tokenizer.json")
    tok = load_tokenizer(tok_path)

    model = KothaGPT(cfg.model)
    # Load weights if checkpoint exists
    ckpt = artifact / "checkpoints" / "best.pt"
    if ckpt.is_file():
        import torch
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model_state = state.get("model_state", state) if isinstance(state, dict) else state
        model.load_state_dict(model_state, strict=False)

    if quant != "none":
        from ml.inference.quant import quantize_model

        bits = 8 if quant == "int8" else 4
        quantize_model(model, bits=bits)

    model.to(device).eval()
    # Atomic swap for hot-reload would be done by caller holding reference
    return model, tok, rec

def hot_reload(model_id: str, old_model: Any, device: str = "cpu"):
    # Simple: load new and return, caller swaps reference atomically
    return load_model(model_id, device=device)

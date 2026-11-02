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

    quant_value = _normalize_quant(quant)
    artifact = (
        Path(rec.artifact_path)
        if rec.artifact_path
        else _REPO_ROOT / "ml" / "pretrain" / "artifacts" / model_id
    )
    if not artifact.is_absolute():
        artifact = _REPO_ROOT / artifact

    from ml.models import KothaGPT
    from ml.tokenizer import load_tokenizer

    cfg = _model_config_for_record(rec, artifact)
    tok_path = _REPO_ROOT / "ml" / "tokenizer" / "artifacts" / "best" / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tok_path}")
    tok = load_tokenizer(tok_path)

    model = KothaGPT(cfg.model)
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
    return model, tok, rec


def hot_reload(model_id: str, old_model: Any, device: str = "cpu", quant: str = "none"):
    # Simple: load new and return, caller swaps reference atomically.
    return load_model(model_id, device=device, quant=quant)

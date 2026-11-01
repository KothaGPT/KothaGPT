"""WS-8 Model loading system — registry-aware, quant-aware, hot-reload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.inference.registry import get_registry

_ALLOWED_QUANT_LEVELS = {"none", "int8", "int4", "bf16"}


def _normalize_quant(quant: str | None) -> str:
    value = (quant or "none").strip().lower()
    if value not in _ALLOWED_QUANT_LEVELS:
        raise ValueError(f"unsupported quant mode {quant!r}; expected one of {sorted(_ALLOWED_QUANT_LEVELS)}")
    return value


def _model_config_for_record(rec, artifact: Path):
    from ml.models import load_config
    from ml.models.config import ModelConfig

    cfg_path = artifact / "config.json" if artifact.is_dir() else Path("ml/configs/small.yaml")
    if cfg_path.is_file():
        try:
            return load_config(cfg_path)
        except Exception:
            pass

    return type(
        "Cfg",
        (),
        {
            "model": ModelConfig(
                vocab_size=698,
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                max_position_embeddings=rec.context_window,
            )
        },
    )()


def load_model(model_id: str = "kothagpt", device: str = "cpu", quant: str = "none"):
    reg = get_registry()
    rec = reg.get(model_id)
    if not rec:
        raise ValueError(f"model {model_id!r} not in registry")

    quant_value = _normalize_quant(quant)
    artifact = Path(rec.artifact_path) if rec.artifact_path else Path(f"ml/pretrain/artifacts/{model_id}")

    from ml.models import KothaGPT
    from ml.tokenizer import load_tokenizer

    cfg = _model_config_for_record(rec, artifact)
    tok_path = Path("ml/tokenizer/artifacts/best/tokenizer.json")
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

    if quant_value == "bf16":
        import torch

        if not hasattr(torch, "bfloat16"):
            raise ValueError("bf16 precision is not supported in this PyTorch build")
        model.to(device=device, dtype=torch.bfloat16)
    elif quant_value != "none":
        from ml.inference.quant import quantize_model

        bits = 8 if quant_value == "int8" else 4
        quantize_model(model, bits=bits)

    model.to(device).eval()
    return model, tok, rec


def hot_reload(model_id: str, old_model: Any, device: str = "cpu", quant: str = "none"):
    # Simple: load new and return, caller swaps reference atomically.
    return load_model(model_id, device=device, quant=quant)

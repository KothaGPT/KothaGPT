"""Atomic, resumable, versioned model checkpoints.

Layout under ``<out>/checkpoints/``::

    step-<N>.pt        full training state (weights + optimizer + scheduler)
    config.json        the run's ``BaseModelConfig`` (canonical JSON)
    metadata.json      digests + counters + tokenizer/data versions

Saves write to a temp file then rename into place so a partially written
checkpoint is never loadable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from ml.models import BaseModelConfig, config_digest, model_digest
from ml.tokenizer import load_tokenizer


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dir_digest(path: Path) -> str:
    """Stable digest over the sorted file list + sizes under ``path``."""
    if not path.exists():
        return "missing"
    entries = sorted((p.name, p.stat().st_size) for p in path.rglob("*") if p.is_file())
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def metadata_for(
    config: BaseModelConfig,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    train_path = Path(config.data.train)
    validation_path = Path(config.data.validation)
    train_digest = _dir_digest(train_path)
    validation_digest = _dir_digest(validation_path)
    run_id = hashlib.sha256(
        json.dumps(
            {
                "config_digest": config_digest(config),
                "data_train": train_digest,
                "data_validation": validation_digest,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    data_version = _data_version(train_path)
    return {
        "run_id": run_id,
        "data_version": data_version,
        "shard_offset": 0,
        "config_digest": config_digest(config),
        "tokenizer_version": tokenizer.type,
        "tokenizer_digest": _sha256_file(Path(tokenizer_path)),
        "data_train_digest": train_digest,
        "data_validation_digest": validation_digest,
    }


def _data_version(train_path: Path) -> str:
    """Corpus version from a tokenized MANIFEST, else the train-dir digest."""
    manifest = train_path / "MANIFEST.json"
    if manifest.is_file():
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))["corpus_version"]
        except (json.JSONDecodeError, KeyError):
            pass
    return _dir_digest(train_path)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: BaseModelConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "metadata": metadata,
    }
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    payload["rng_state"] = rng_state
    return payload


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_text(content: str, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_checkpoint(
    out_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: BaseModelConfig,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write a full training-state checkpoint."""
    out_dir = Path(out_dir)
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    config_text = (
        json.dumps(
            {
                "model": config.model.to_dict(),
                "training": config.training.to_dict(),
                "data": config.data.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    _atomic_write_text(config_text, out_dir / "config.json")
    _atomic_write_text(json.dumps(metadata, indent=2) + "\n", out_dir / "metadata.json")

    final_path = checkpoints / f"step-{step:07d}.pt"
    _atomic_save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            metadata=metadata,
        ),
        final_path,
    )
    _prune(checkpoints, keep=config.training.keep_last)
    return final_path


def save_best_checkpoint(
    out_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: BaseModelConfig,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write the best-validation checkpoint (kept across pruning)."""
    checkpoints = Path(out_dir) / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints / "best.pt"
    _atomic_save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            metadata=metadata,
        ),
        best_path,
    )
    return best_path


def _prune(checkpoints: Path, keep: int) -> None:
    existing = sorted(checkpoints.glob("step-*.pt"))
    for stale in existing[:-keep]:
        stale.unlink()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def step_of(path: str | Path) -> int:
    return load_checkpoint(path)["step"]


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    matches = sorted(Path(out_dir).glob("checkpoints/step-*.pt"))
    return matches[-1] if matches else None


def resume(
    out_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: BaseModelConfig,
) -> int:
    """Restore the latest checkpoint into model/optimizer/scheduler in place.

    Returns the step to resume from (the count already completed).
    """
    path = latest_checkpoint(out_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint to resume in {out_dir}")
    state = load_checkpoint(path)
    saved = state["config"]
    if model_digest(saved) != model_digest(config):
        raise ValueError(
            f"checkpoint model config mismatch: {path} "
            f"(saved {model_digest(saved)} vs {model_digest(config)})"
        )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    if "cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state["step"]

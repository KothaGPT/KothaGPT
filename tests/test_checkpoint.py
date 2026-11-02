"""Tests for checkpoint save/load/resume semantics (WS-6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.models import KothaGPT, ModelConfig
from ml.trainer import (
    CausalLMDataset,
    build_blocks,
    group_parameters,
    latest_checkpoint,
    load_checkpoint,
    resume,
    save_checkpoint,
    step_of,
    train,
)
from tests.conftest import make_corpus


def test_save_and_load_roundtrip(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "run",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=7,
        config=config,
        metadata={"config_digest": "abc"},
    )
    assert path.name == "step-0000007.pt"
    assert step_of(path) == 7
    state = load_checkpoint(path)
    assert state["step"] == 7
    assert state["config"].model.hidden_size == config.model.hidden_size


def test_latest_checkpoint_returns_newest(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (1, 2, 3):
        save_checkpoint(
            tmp_path / "run",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=step,
            config=config,
            metadata={},
        )
    latest = latest_checkpoint(tmp_path / "run")
    assert latest is not None
    assert step_of(latest) == 3


def test_resume_restores_step(config, tokenizer, tmp_path: Path) -> None:
    model = KothaGPT(config.model)
    out = tmp_path / "run"
    train(
        model,
        config,
        CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8)),
        None,
        out_dir=out,
        device="cpu",
    )
    restored = KothaGPT(config.model)
    restored_opt = torch.optim.AdamW(
        group_parameters(restored, weight_decay=config.training.weight_decay),
        lr=1e-3,
    )
    step = resume(out, model=restored, optimizer=restored_opt, scheduler=None, config=config)
    assert step == 4
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a.detach(), b.detach())


def test_resume_continues_training(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    out = tmp_path / "run"
    model = KothaGPT(config.model)
    first = train(model, config, dataset, None, out_dir=out, device="cpu")
    assert first["step"] == 4

    continued = config.model
    resumed_model = KothaGPT(continued)
    resumed_config = config
    second = train(
        resumed_model,
        resumed_config,
        dataset,
        None,
        out_dir=out,
        device="cpu",
        max_steps=6,
    )
    assert second["step"] == 6


def test_resume_reproduces_next_step_loss(config, tokenizer, tmp_path: Path) -> None:
    """Fresh-process resume must match a from-scratch run at the same step."""
    from dataclasses import replace

    from ml.models import BaseModelConfig

    corpus = make_corpus(tmp_path, docs=32)
    dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    # The LR schedule is always driven by the config's max_steps (8), while
    # the max_steps override only caps the run — so a resume mid-run stays on
    # the same cosine schedule as a from-scratch run.
    cfg = BaseModelConfig(
        model=config.model,
        training=replace(config.training, max_steps=8),
        data=config.data,
    )

    seed_run = tmp_path / "seed"
    torch.manual_seed(0)
    train(
        KothaGPT(cfg.model),
        cfg,
        dataset,
        None,
        out_dir=seed_run,
        device="cpu",
        max_steps=4,
    )

    resumed_out = tmp_path / "resumed"
    torch.manual_seed(0)
    train(
        KothaGPT(cfg.model),
        cfg,
        dataset,
        None,
        out_dir=resumed_out,
        device="cpu",
        resume_from=seed_run,
        max_steps=6,
    )

    fresh_out = tmp_path / "fresh"
    torch.manual_seed(0)
    train(
        KothaGPT(cfg.model),
        cfg,
        dataset,
        None,
        out_dir=fresh_out,
        device="cpu",
        max_steps=6,
    )

    def loss_at(out: Path, step: int) -> float:
        rows = [
            json.loads(line)
            for line in (out / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        ]
        return next(r["loss"] for r in rows if r["step"] == step)

    assert loss_at(resumed_out, 5) == loss_at(fresh_out, 5)
    assert loss_at(resumed_out, 6) == loss_at(fresh_out, 6)


def test_metadata_has_run_identity(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    out = tmp_path / "run"
    train(KothaGPT(config.model), config, dataset, None, out_dir=out, device="cpu")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"]
    assert metadata["data_version"]
    assert metadata["shard_offset"] == 0
    assert metadata["config_digest"]
    assert metadata["tokenizer_digest"]


def test_resume_model_mismatch_raises(config, tokenizer, tmp_path: Path) -> None:
    dataset = CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8))
    out = tmp_path / "run"
    model = KothaGPT(config.model)
    train(model, config, dataset, None, out_dir=out, device="cpu")

    mismatched = ModelConfig(**{**config.model.to_dict(), "hidden_size": 48})
    wrong = KothaGPT(mismatched)
    opt = torch.optim.AdamW(wrong.parameters(), lr=1e-3)

    from ml.models import BaseModelConfig

    mismatched_cfg = BaseModelConfig(model=mismatched, training=config.training, data=config.data)
    with pytest.raises(ValueError, match="model config mismatch"):
        resume(out, model=wrong, optimizer=opt, scheduler=None, config=mismatched_cfg)


def test_checkpoint_pruning_keeps_last(config, tokenizer, tmp_path: Path) -> None:
    from dataclasses import replace

    dataset = CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=8), tokenizer, block_size=8))
    pruned_cfg = replace(config, training=replace(config.training, keep_last=1))
    model = KothaGPT(pruned_cfg.model)
    out = tmp_path / "run"
    train(model, pruned_cfg, dataset, None, out_dir=out, device="cpu")
    checkpoints = sorted((out / "checkpoints").glob("step-*.pt"))
    assert len(checkpoints) == 1
    assert step_of(checkpoints[0]) == 4


def test_best_checkpoint_survives_pruning_and_matches_metadata(
    config, tokenizer, tmp_path: Path
) -> None:
    from dataclasses import replace

    dataset = CausalLMDataset(build_blocks(make_corpus(tmp_path, docs=16), tokenizer, block_size=8))
    pruned_cfg = replace(
        config,
        training=replace(config.training, keep_last=1, eval_interval=1, eval_batches=1),
    )
    out = tmp_path / "run"
    train(KothaGPT(pruned_cfg.model), pruned_cfg, dataset, dataset, out_dir=out, device="cpu")

    best = out / "checkpoints" / "best.pt"
    assert best.is_file()
    state = load_checkpoint(best)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert state["metadata"] == metadata
    assert state["step"] >= 1
    assert len(list((out / "checkpoints").glob("step-*.pt"))) == 1

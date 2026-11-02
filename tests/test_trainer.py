"""Tests for the training loop, dataset, scheduler, and monitoring (WS-2/8/9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.models import BaseModelConfig, KothaGPT, ModelConfig, TrainingConfig
from ml.trainer import (
    CausalLMDataset,
    build_blocks,
    build_scheduler,
    evaluate,
    group_parameters,
    sample_text,
    train,
)
from ml.trainer.checkpoint import latest_checkpoint, load_checkpoint
from tests.conftest import make_corpus


def test_build_blocks_shapes_and_shifts(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=4)
    blocks = build_blocks(corpus, tokenizer, block_size=8)
    assert blocks.ndim == 2
    assert blocks.shape[1] == 8
    ds = CausalLMDataset(blocks)
    x, y = ds[0]
    assert x.shape == (7,)
    assert y.shape == (7,)
    assert torch.equal(x, blocks[0][:-1])
    assert torch.equal(y, blocks[0][1:])


def test_group_parameters_excludes_bias_and_norms(config) -> None:
    model = KothaGPT(config.model)
    groups = group_parameters(model, weight_decay=0.1)
    decay, no_decay = groups[0]["params"], groups[1]["params"]
    assert decay
    assert no_decay
    all_names = dict(model.named_parameters())
    decay_ids = {id(p) for p in decay}
    for name, param in all_names.items():
        if name.endswith(".bias") or "norm" in name or "embed_tokens" in name:
            assert id(param) not in decay_ids, name


def test_scheduler_warmup_and_decay(config) -> None:
    model = KothaGPT(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    sched = build_scheduler(optimizer, config.training)
    initial = optimizer.param_groups[0]["lr"]
    sched.step()
    assert optimizer.param_groups[0]["lr"] > 0
    sched.step()
    assert optimizer.param_groups[0]["lr"] <= initial


def test_train_writes_history_and_checkpoint(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    val_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    model = KothaGPT(config.model)

    out = tmp_path / "run"
    result = train(
        model,
        config,
        train_dataset,
        val_dataset,
        out_dir=out,
        device="cpu",
    )
    assert result["step"] == 4
    lines = [
        json.loads(l)
        for l in (out / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert any("loss" in line for line in lines)
    assert any("lr" in line for line in lines)
    assert (out / "lr_curve.csv").exists()
    assert (out / "lr_curve.svg").exists()
    assert latest_checkpoint(out) is not None


def test_train_loss_decreases(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=16)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    model = KothaGPT(config.model)

    out = tmp_path / "run"
    train(model, config, train_dataset, None, out_dir=out, device="cpu")
    history = [
        json.loads(line)
        for line in (out / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    losses = [entry["loss"] for entry in history]
    assert len(losses) == 4
    assert losses[-1] < losses[0]


def test_evaluate_and_sample(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    model = KothaGPT(config.model)
    metrics = evaluate(model, train_dataset, device="cpu")
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1
    text = sample_text(model, tokenizer, "বাংলা", max_new_tokens=2, device="cpu")
    assert isinstance(text, str)


def test_checkpoint_metadata_saved(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=8)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    model = KothaGPT(config.model)
    out = tmp_path / "run"
    train(model, config, train_dataset, None, out_dir=out, device="cpu")
    state = load_checkpoint(latest_checkpoint(out))
    assert state["step"] == 4
    assert state["metadata"]["config_digest"]
    assert (out / "config.json").exists()
    assert (out / "metadata.json").exists()


def test_best_checkpoint_and_samples(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=16)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    val_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    model = KothaGPT(config.model)
    out = tmp_path / "run"
    train(
        model,
        config,
        train_dataset,
        val_dataset,
        out_dir=out,
        device="cpu",
        tokenizer=tokenizer,
    )
    assert (out / "checkpoints" / "best.pt").exists()
    assert (out / "samples" / "step-0000004.txt").exists()
    best = load_checkpoint(out / "checkpoints" / "best.pt")
    assert best["step"] >= 1


def test_identical_seeds_identical_val_loss(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=32)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    val_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))

    def run() -> float:
        model = KothaGPT(config.model)
        train(
            model,
            config,
            train_dataset,
            val_dataset,
            out_dir=tmp_path / "r",
            device="cpu",
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "r" / "history.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        return next(r["val_ppl"] for r in rows if "val_ppl" in r)

    first = run()
    second = run()
    assert first == second


def test_trend_guard_aborts_on_divergence(config, tokenizer, tmp_path: Path, monkeypatch) -> None:
    from ml.trainer.loop import TrainingDiverged

    corpus = make_corpus(tmp_path, docs=64)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    cfg = BaseModelConfig(
        model=config.model,
        training=TrainingConfig(
            batch_size=2,
            max_steps=10,
            mixed_precision="none",
            trend_guard_patience=4,
            trend_guard_action="abort",
        ),
        data=config.data,
    )
    model = KothaGPT(cfg.model)

    # Force a flat (never-improving) loss so the guard must fire.
    def _flat_loss(self, input_ids, labels=None, **kwargs):
        return {"loss": torch.tensor(999.0, requires_grad=True)}

    monkeypatch.setattr(model, "forward", _flat_loss.__get__(model, KothaGPT))
    with pytest.raises(TrainingDiverged):
        train(model, cfg, train_dataset, None, out_dir=tmp_path / "div", device="cpu")


def test_train_tiny_hidden_size(config, tokenizer, tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, docs=4)
    train_dataset = CausalLMDataset(build_blocks(corpus, tokenizer, block_size=8))
    tiny = BaseModelConfig(
        model=ModelConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            max_position_embeddings=8,
        ),
        training=TrainingConfig(batch_size=2, max_steps=2, mixed_precision="none"),
        data=config.data,
    )
    model = KothaGPT(tiny.model)
    result = train(model, tiny, train_dataset, None, out_dir=tmp_path / "tiny", device="cpu")
    assert result["step"] == 2

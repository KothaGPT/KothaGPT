"""WS-10 — Learning-rate scheduling tests (ml/trainer/scheduler.py)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from ml.models import TrainingConfig
from ml.trainer.scheduler import (
    build_scheduler,
    lr_multiplier,
    schedule_points,
    write_schedule_curve,
)


def _training(**overrides) -> TrainingConfig:
    base = {
        "learning_rate": 1e-3,
        "max_steps": 100,
        "warmup_steps": 10,
        "min_lr": None,
        "lr_schedule": "cosine",
    }
    base.update(overrides)
    cfg = TrainingConfig(**base)
    cfg.validate()
    return cfg


def _run_schedule(training: TrainingConfig) -> list[float]:
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = torch.optim.AdamW(params, lr=training.learning_rate)
    sched = build_scheduler(optimizer, training)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(training.max_steps):
        sched.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_warmup_ramps_learning_rate() -> None:
    training = _training()
    lrs = _run_schedule(training)
    assert lrs[1] < lrs[10]
    assert math.isclose(lrs[10], training.learning_rate, rel_tol=1e-4)


def test_cosine_decays_to_min_lr() -> None:
    training = _training(min_lr=1e-4)
    lrs = _run_schedule(training)
    assert math.isclose(lrs[-1], training.min_lr, rel_tol=1e-6)


def test_default_floor_is_lr_over_ten() -> None:
    training = _training()
    lrs = _run_schedule(training)
    assert math.isclose(lrs[-1], training.learning_rate / 10, rel_tol=1e-6)


def test_linear_decay_is_linear() -> None:
    training = _training(lr_schedule="linear", min_lr=1e-4)
    lrs = _run_schedule(training)
    mid = training.warmup_steps + (training.max_steps - training.warmup_steps) // 2
    expected_mid = (training.learning_rate + training.min_lr) / 2
    assert math.isclose(lrs[mid], expected_mid, rel_tol=1e-3)
    assert math.isclose(lrs[-1], training.min_lr, rel_tol=1e-6)


def test_curve_matches_scheduler() -> None:
    training = _training(lr_schedule="linear", min_lr=2e-4)
    actual = _run_schedule(training)
    predicted = [training.learning_rate * lr_multiplier(s, training) for s in range(len(actual))]
    for a, p in zip(actual, predicted):
        assert math.isclose(a, p, rel_tol=1e-4)


def test_schedule_points_end_at_max_steps() -> None:
    points = schedule_points(_training(max_steps=5000), max_points=100)
    assert len(points) <= 101
    assert points[0][0] == 0
    assert points[-1][0] == 5000


def test_write_schedule_curve_files(tmp_path: Path) -> None:
    svg = write_schedule_curve(_training(), tmp_path)
    assert svg.exists()
    assert (tmp_path / "lr_curve.csv").exists()
    rows = list(
        csv.DictReader((tmp_path / "lr_curve.csv").read_text(encoding="utf-8").splitlines())
    )
    assert rows[0]["step"] == "0"
    text = svg.read_text(encoding="utf-8")
    assert "<svg" in text and "<polyline" in text


def test_config_round_trips_via_yaml(tmp_path: Path) -> None:
    from ml.models import load_config

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "training:\n"
        "  learning_rate: 2.0e-4\n"
        "  max_steps: 200\n"
        "  warmup_steps: 20\n"
        "  min_lr: 4.0e-5\n"
        "  lr_schedule: linear\n"
        "model:\n"
        "  vocab_size: 128\n"
        "data:\n"
        "  train: /tmp/none\n"
        "  tokenizer_path: /tmp/none/tokenizer.json\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg_path)
    assert loaded.training.warmup_steps == 20
    assert loaded.training.min_lr == 4.0e-5
    assert loaded.training.lr_schedule == "linear"
    lrs = _run_schedule(loaded.training)
    assert math.isclose(lrs[-1], 4.0e-5, rel_tol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_lr": -1e-5},
        {"min_lr": 2e-3},  # above learning_rate (1e-3)
        {"lr_schedule": "exponential"},
        {"warmup_steps": -1},
        {"warmup_steps": 200},  # above max_steps (100)
    ],
)
def test_invalid_schedule_config_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _training(**overrides)

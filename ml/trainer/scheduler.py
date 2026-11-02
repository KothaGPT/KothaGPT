"""Learning-rate schedules for training.

Supports cosine and linear decay over ``training.max_steps`` with a linear
warmup over ``training.warmup_steps`` and a configurable floor at
``training.min_lr`` (defaults to ``learning_rate / 10``). The schedule curve
(``lr`` vs ``step``) can be written to the run dir as a CSV plus a small,
dependency-free SVG plot via :func:`write_schedule_curve`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ml.models import TrainingConfig


def min_lr_fraction(training: TrainingConfig) -> float:
    """Floor LR as a fraction of ``learning_rate`` (default ``0.1``)."""
    if training.min_lr is not None:
        return training.min_lr / training.learning_rate
    return 0.1


def lr_multiplier(step: int, training: TrainingConfig) -> float:
    """Schedule multiplier for a given macro-step (shared by build/curve)."""
    if training.warmup_steps > 0 and step < training.warmup_steps:
        return (step + 1) / training.warmup_steps
    progress = (step - training.warmup_steps) / max(training.max_steps - training.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    floor = min_lr_fraction(training)
    if training.lr_schedule == "linear":
        return 1.0 - (1.0 - floor) * progress
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: Optimizer,
    training: TrainingConfig,
) -> LambdaLR:
    """Warmup then cosine or linear decay to ``min_lr`` at ``max_steps``."""
    return LambdaLR(optimizer, lambda step: lr_multiplier(step, training))


def schedule_points(training: TrainingConfig, max_points: int = 1000) -> list[tuple[int, float]]:
    """Sample ``(step, lr)`` along the full schedule (bounded to ``max_points``)."""
    steps = range(training.max_steps + 1)
    stride = max(1, (training.max_steps + 1) // max_points)
    points = [
        (step, training.learning_rate * lr_multiplier(step, training)) for step in steps[::stride]
    ]
    if points[-1][0] != training.max_steps:
        points.append(
            (
                training.max_steps,
                training.learning_rate * lr_multiplier(training.max_steps, training),
            )
        )
    return points


def write_schedule_curve(training: TrainingConfig, out_dir: str | Path) -> Path:
    """Write ``lr_curve.csv`` + ``lr_curve.svg`` describing the schedule."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = schedule_points(training)

    csv_path = out_dir / "lr_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr"])
        writer.writerows(points)

    svg_path = out_dir / "lr_curve.svg"
    svg_path.write_text(_svg_plot(points), encoding="utf-8")
    return svg_path


def _svg_plot(points: list[tuple[int, float]], width: int = 640, height: int = 320) -> str:
    """Render a tiny dependency-free line chart as an SVG string."""
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    steps = [p[0] for p in points]
    lrs = [p[1] for p in points]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(lrs), max(lrs)
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-12)

    def px(step: int) -> float:
        return pad_l + (step - x_min) / x_span * plot_w

    def py(lr: float) -> float:
        return pad_t + (1 - (lr - y_min) / y_span) * plot_h

    path = "M " + " L ".join(f"{px(s):.1f},{py(l):.1f}" for s, l in points)
    grid = ""
    for frac in (0.25, 0.5, 0.75):
        y = pad_t + frac * plot_h
        grid += f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width - pad_r}' y2='{y:.1f}' stroke='#e5e7eb' stroke-width='1'/>"
    axes = (
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{height - pad_b}' stroke='#9ca3af' stroke-width='1'/>"
        f"<line x1='{pad_l}' y1='{height - pad_b}' x2='{width - pad_r}' y2='{height - pad_b}' stroke='#9ca3af' stroke-width='1'/>"
    )
    y_label_max = f"<text x='{pad_l - 8}' y='{pad_t + 4}' text-anchor='end' font-size='10' fill='#6b7280'>{y_max:.2e}</text>"
    y_label_min = f"<text x='{pad_l - 8}' y='{height - pad_b + 4}' text-anchor='end' font-size='10' fill='#6b7280'>{y_min:.2e}</text>"
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"{grid}{axes}{y_label_max}{y_label_min}"
        f"<polyline points='{path[2:]}' fill='none' stroke='#2563eb' stroke-width='2' stroke-linejoin='round'/>"
        f"<text x='{width / 2}' y='{height - 8}' text-anchor='middle' font-size='11' fill='#374151'>step</text>"
        f"</svg>"
    )


def group_parameters(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into weight-decay and no-weight-decay groups.

    Embeddings, biases, and norm weights are excluded from weight decay.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name or "embed_tokens" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

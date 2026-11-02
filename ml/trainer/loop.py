"""The training loop: mixed precision, gradient accumulation, distributed."""

from __future__ import annotations

import math
import resource
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ml.models import BaseModelConfig, KothaGPT, TrainingConfig

from .checkpoint import (
    latest_checkpoint,
    metadata_for,
    resume,
    save_best_checkpoint,
    save_checkpoint,
    step_of,
)
from .dataset import CausalLMDataset
from .distributed import DistributedConfig, all_reduce_mean, is_main_rank, unwrap_module, wrap_model
from .evaluate import sample_text
from .monitor import Monitor
from .scheduler import build_scheduler, group_parameters, write_schedule_curve


class TrainingDiverged(RuntimeError):
    """Raised when the loss-trend guard aborts a run (trend_guard_action=abort)."""


def _autocast_context(training: TrainingConfig, device: str):
    if training.mixed_precision in ("bf16", "fp16") and device.startswith("cuda"):
        dtype = torch.bfloat16 if training.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _make_optimizer(model: torch.nn.Module, training: TrainingConfig):
    params = group_parameters(model, training.weight_decay)
    return torch.optim.AdamW(
        params,
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
        weight_decay=training.weight_decay,
    )


def _collate_batch(dataset, indices: list[int]):
    """Stack per-sample ``(input_ids, labels)`` pairs into one batch."""
    parts = [dataset[i] for i in indices]
    return torch.stack([p[0] for p in parts]), torch.stack([p[1] for p in parts])


def _peak_mem_mb(device: str) -> float:
    """Peak resident memory for the step, in MB (GPU alloc or process RSS)."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device) / 1e6
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    return rss / 1024.0


def _validate_step(
    model: torch.nn.Module,
    validation: DataLoader,
    training: TrainingConfig,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    seen = 0
    cap = training.eval_batches
    with torch.no_grad(), _autocast_context(training, device):
        for input_ids, labels in validation:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            total += model(input_ids=input_ids, labels=labels)["loss"].item()
            seen += 1
            if seen >= cap:
                break
    return total / max(seen, 1)


def train(
    model: KothaGPT,
    config: BaseModelConfig,
    train_dataset: CausalLMDataset,
    validation_dataset: CausalLMDataset | None,
    *,
    out_dir: str | Path,
    device: str = "cpu",
    start_step: int = 0,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    dist: DistributedConfig | None = None,
    tokenizer=None,
) -> dict:
    dist = dist or DistributedConfig(distributed=False).effective()
    rank = dist.rank or 0
    world_size = dist.world_size or 1

    training = config.training
    # max_steps override only caps the run; the LR schedule always uses the
    # config's max_steps so resumed runs stay on the same cosine schedule.
    stop_step = training.max_steps
    if max_steps is not None:
        stop_step = max_steps

    torch.manual_seed(training.seed + rank)
    torch.cuda.manual_seed_all(training.seed + rank)

    model.to(device)
    model.train()
    model.gradient_checkpointing = training.gradient_checkpointing

    optimizer = _make_optimizer(model, training)
    scheduler = build_scheduler(optimizer, training)

    if is_main_rank(rank):
        write_schedule_curve(training, out_dir)

    if resume_from is not None and latest_checkpoint(resume_from) is not None:
        start_step = resume(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )

    model = wrap_model(model, dist, device)
    model.train()

    monitor = Monitor(out_dir, rank=rank)
    metadata = metadata_for(config, config.data.tokenizer_path)

    micro_batches = training.gradient_accumulation_steps

    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training.shuffle,
            seed=training.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=training.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
        )
        steps_per_epoch = 0
        macro = 0
    else:
        # Deterministic, index-driven batches: the permutation for an epoch is
        # derived purely from (seed, rank, epoch), so resuming at any global
        # step reproduces the exact batch a from-scratch run would consume.
        train_loader = None
        n_train = len(train_dataset)
        usable = (n_train // training.batch_size) * training.batch_size
        macro = micro_batches * training.batch_size
        steps_per_epoch = usable // macro
    validation_loader = (
        DataLoader(validation_dataset, batch_size=training.batch_size, shuffle=False)
        if validation_dataset is not None
        else None
    )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=training.mixed_precision == "fp16")
        if device.startswith("cuda")
        else None
    )

    global_step = start_step
    accumulated_loss = 0.0
    accumulated_tokens = 0
    last_loss = 0.0
    epoch = 0
    step_start = time.monotonic()
    smoothed_loss = 0.0
    last_smoothed: float | None = None
    flat_steps = 0
    best_val_ppl = float("inf")

    def finalize_step() -> None:
        nonlocal accumulated_loss, accumulated_tokens, last_loss, global_step, step_start
        nonlocal smoothed_loss, last_smoothed, flat_steps, best_val_ppl
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
        grad_norm_val = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        reported_loss = all_reduce_mean(
            torch.tensor([accumulated_loss], device=device), dist
        ).item()
        last_loss = reported_loss

        now = time.monotonic()
        dt = max(now - step_start, 1e-9)
        tokens_per_sec = accumulated_tokens / dt
        step_tokens = accumulated_tokens
        accumulated_loss = 0.0
        accumulated_tokens = 0
        step_start = now

        lr = optimizer.param_groups[0]["lr"]
        monitor.log(
            global_step,
            {
                "loss": reported_loss,
                "lr": lr,
                "tokens": step_tokens,
                "tokens_per_sec": tokens_per_sec,
                "grad_norm": grad_norm_val,
                "peak_mem_mb": _peak_mem_mb(device),
            },
        )

        if training.trend_guard_patience > 0 and is_main_rank(rank):
            smoothed_loss = (
                0.9 * smoothed_loss + 0.1 * reported_loss if smoothed_loss else reported_loss
            )
            if last_smoothed is not None:
                if smoothed_loss >= last_smoothed - 1e-6:
                    flat_steps += 1
                else:
                    flat_steps = 0
                if flat_steps >= training.trend_guard_patience:
                    msg = f"smoothed loss {smoothed_loss:.4f} not improving for {flat_steps} steps"
                    if training.trend_guard_action == "abort":
                        raise TrainingDiverged(msg)
                    print(f"[trend-guard] WARNING: {msg}", flush=True)
                    flat_steps = 0
            last_smoothed = smoothed_loss

        if (
            is_main_rank(rank)
            and global_step % training.eval_interval == 0
            and validation_loader is not None
        ):
            val_loss = _validate_step(model, validation_loader, training, device)
            val_ppl = math.exp(min(val_loss, 20))
            monitor.log(global_step, {"val_loss": val_loss, "val_ppl": val_ppl})
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                save_best_checkpoint(
                    out_dir,
                    model=unwrap_module(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    config=config,
                    metadata=metadata,
                )
                print(
                    f"step {global_step}: new best val_ppl={val_ppl:.4f} (checkpoints/best.pt)",
                    flush=True,
                )
            if tokenizer is not None:
                samples_dir = Path(out_dir) / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                text = sample_text(model, tokenizer, "বাংলা", max_new_tokens=16, device=device)
                (samples_dir / f"step-{global_step:07d}.txt").write_text(
                    text + "\n", encoding="utf-8"
                )

        if is_main_rank(rank) and global_step % training.save_interval == 0:
            save_checkpoint(
                out_dir,
                model=unwrap_module(model),
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                config=config,
                metadata=metadata,
            )

    def micro_step(input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal accumulated_loss, accumulated_tokens
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with _autocast_context(training, device):
            loss = model(input_ids=input_ids, labels=labels)["loss"] / micro_batches
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accumulated_loss += loss.item()
        accumulated_tokens += input_ids.numel()

    def epoch_perm(epoch: int) -> torch.Tensor:
        if not training.shuffle:
            return torch.arange(len(train_dataset))
        g = torch.Generator()
        g.manual_seed(training.seed + rank + epoch)
        return torch.randperm(len(train_dataset), generator=g)

    while global_step < stop_step:
        if world_size > 1:
            epoch += 1
            sampler.set_epoch(epoch)
            any_batch = False
            for micro_count, (input_ids, labels) in enumerate(train_loader, start=1):
                any_batch = True
                micro_step(input_ids, labels)
                if micro_count % micro_batches == 0:
                    finalize_step()
                    if global_step >= stop_step:
                        break
            if not any_batch:
                break
            if accumulated_loss > 0:
                finalize_step()
        else:
            if steps_per_epoch == 0:
                break
            epoch = global_step // steps_per_epoch
            perm = epoch_perm(epoch)
            off = (global_step % steps_per_epoch) * macro
            for mb in range(micro_batches):
                indices = perm[
                    off + mb * training.batch_size : off + (mb + 1) * training.batch_size
                ].tolist()
                micro_step(*_collate_batch(train_dataset, indices))
            finalize_step()

    latest = latest_checkpoint(out_dir) if is_main_rank(rank) else None
    if is_main_rank(rank) and (latest is None or step_of(latest) < global_step):
        save_checkpoint(
            out_dir,
            model=unwrap_module(model),
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            config=config,
            metadata=metadata,
        )
    return {"step": global_step, "loss": last_loss}

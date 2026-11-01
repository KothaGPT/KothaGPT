"""Distributed training support: process-group init, DDP/FSDP wrapping.

CPU testing uses the ``gloo`` backend (no launcher required); GPUs use
``nccl``. FSDP requires CUDA. Rank 0 is the only rank that logs, evaluates, and
writes checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedConfig:
    """Process-group settings; all fields are auto-derived from env by default."""

    world_size: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    backend: str | None = None
    use_fsdp: bool = False
    distributed: bool = False
    init_method: str | None = None

    def effective(self) -> DistributedConfig:
        if not self.distributed:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            return self
        if self.world_size is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.rank is None:
            self.rank = int(os.environ.get("RANK", "0"))
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        return self


def init_distributed(cfg: DistributedConfig) -> DistributedConfig:
    """Initialize the process group; a no-op when not distributed."""
    cfg = cfg.effective()
    if not cfg.distributed or cfg.world_size <= 1:
        return cfg
    backend = cfg.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    init_method = cfg.init_method or f"tcp://{cfg.master_addr}:{cfg.master_port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )
    return cfg


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(tensor: torch.Tensor, cfg: DistributedConfig) -> torch.Tensor:
    """Average a scalar tensor across ranks."""
    if not (cfg.distributed and dist.is_initialized() and cfg.world_size > 1):
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / cfg.world_size


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, peeling off DDP/FSDP wrappers."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        return model.module
    return model


def wrap_model(
    model: torch.nn.Module,
    cfg: DistributedConfig,
    device: str,
) -> torch.nn.Module:
    """Wrap ``model`` for distributed training (DDP default, FSDP optional).

    The model must already be moved to ``device``. Optimizers built on the raw
    model parameters keep working because DDP/FSDP share the underlying
    parameter tensors.
    """
    if not cfg.distributed or cfg.world_size <= 1:
        return model
    if not dist.is_initialized():
        raise RuntimeError("process group not initialized; call init_distributed first")
    if cfg.use_fsdp:
        if not device.startswith("cuda"):
            raise RuntimeError("FSDP requires CUDA; use DDP on CPU or add GPUs")
        from ml.models.blocks import TransformerBlock
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerBlock},
        )
        return FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            use_orig_params=True,
            device_id=cfg.local_rank,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank] if device.startswith("cuda") else None,
        find_unused_parameters=False,
    )

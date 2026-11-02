"""Training CLI: ``python -m ml.trainer.cli run --config ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.models import KothaGPT, load_config
from ml.tokenizer import load_tokenizer

from .dataset import CausalLMDataset, ShardedMemmapDataset, build_blocks
from .distributed import DistributedConfig, destroy_distributed, init_distributed, unwrap_module
from .evaluate import sample_text
from .loop import train


def _is_tokenized(path: str) -> bool:
    return (Path(path) / "MANIFEST.json").is_file()


def _resolve_data_path(path: str) -> Path:
    """Resolve a ``data/tokenized/CURRENT`` pointer to its concrete dir."""
    p = Path(path)
    if p.is_file() and p.name == "CURRENT":
        return p.parent / p.read_text(encoding="utf-8").strip()
    return p


def _build_datasets(data, block_size: int, dist: DistributedConfig) -> tuple:
    """Build train/validation datasets, preferring the WS-1 tokenized shards.

    When ``data.train`` points at a tokenized corpus (has MANIFEST.json) a
    memmap-backed :class:`ShardedMemmapDataset` is used with per-rank shard
    assignment; otherwise the eager ``build_blocks`` path is kept.
    """

    rank = dist.rank if dist.distributed else None
    world_size = dist.world_size if dist.distributed else None

    train_dir = _resolve_data_path(data.train)
    val_dir = _resolve_data_path(data.validation)

    if _is_tokenized(str(train_dir)):
        train_dataset = ShardedMemmapDataset(
            train_dir, split="train", rank=rank, world_size=world_size
        )
    else:
        tokenizer = load_tokenizer(data.tokenizer_path)
        train_dataset = CausalLMDataset(build_blocks(train_dir, tokenizer, block_size))

    validation_dataset = None
    if val_dir.exists():
        try:
            if _is_tokenized(str(val_dir)):
                validation_dataset = ShardedMemmapDataset(val_dir, split="validation")
            else:
                tokenizer = load_tokenizer(data.tokenizer_path)
                validation_dataset = CausalLMDataset(build_blocks(val_dir, tokenizer, block_size))
        except (FileNotFoundError, ValueError):
            validation_dataset = None
    return train_dataset, validation_dataset


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    if args.tokenizer:
        config.data.tokenizer_path = args.tokenizer

    dist = DistributedConfig(
        world_size=args.world_size,
        rank=args.rank,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        use_fsdp=args.use_fsdp,
        distributed=args.distributed,
        init_method=args.init_method,
    )
    dist = init_distributed(dist)

    # Resolve data/tokenized/CURRENT pointers so config.data carries concrete
    # paths everywhere downstream (datasets, metadata, resume).
    config.data.train = str(_resolve_data_path(config.data.train))
    config.data.validation = str(_resolve_data_path(config.data.validation))

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    block_size = config.model.max_position_embeddings

    train_dataset, validation_dataset = _build_datasets(config.data, block_size, dist)
    if hasattr(train_dataset, "block_size") and train_dataset.block_size != block_size:
        raise ValueError(
            f"tokenized corpus block_size={train_dataset.block_size} != "
            f"model.max_position_embeddings={block_size}; tokenize shards at "
            "--block-size matching the model context"
        )

    model = KothaGPT(config.model, gradient_checkpointing=config.training.gradient_checkpointing)

    resume_from = args.out if args.resume else None

    result = train(
        model,
        config,
        train_dataset,
        validation_dataset,
        out_dir=args.out,
        device=args.device,
        resume_from=resume_from,
        max_steps=args.max_steps,
        dist=dist,
        tokenizer=tokenizer,
    )

    if (dist.rank or 0) == 0:
        print(json.dumps(result))
        if validation_dataset is not None:
            text = sample_text(
                unwrap_module(model),
                tokenizer,
                "বাংলা ভাষা",
                max_new_tokens=8,
                temperature=1.0,
                device=args.device,
            )
            print(f"sample: {text}")
    destroy_distributed()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.trainer", description="KothaGPT base-model training")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a training job")
    run.add_argument("--config", required=True, help="path to YAML config")
    run.add_argument("--tokenizer", default=None, help="override tokenizer artifact path")
    run.add_argument("--out", default="ml/pretrain/artifacts/run", help="run output directory")
    run.add_argument("--device", default="cpu", help="device (cpu or cuda[:N])")
    run.add_argument("--max-steps", type=int, default=None, help="override max steps")
    run.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    run.add_argument("--distributed", action="store_true", help="enable distributed training")
    run.add_argument(
        "--world-size", type=int, default=None, help="number of ranks (default from WORLD_SIZE)"
    )
    run.add_argument(
        "--rank", type=int, default=None, help="rank of this process (default from RANK)"
    )
    run.add_argument(
        "--local-rank", type=int, default=None, help="local rank (default from LOCAL_RANK)"
    )
    run.add_argument("--master-addr", default="127.0.0.1", help="master address")
    run.add_argument("--master-port", default="29500", help="master port")
    run.add_argument("--backend", default=None, help="distributed backend (nccl|gloo)")
    run.add_argument("--init-method", default=None, help="torch.distributed init method URL")
    run.add_argument(
        "--use-fsdp", action="store_true", help="use FSDP instead of DDP (requires CUDA)"
    )
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

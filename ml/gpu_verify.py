"""Verify the local PyTorch CUDA environment and run a tiny GPU smoke test."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any


def inspect_environment(torch_module: Any) -> dict[str, Any]:
    """Return a serializable summary of CUDA and distributed capabilities."""
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    count = int(cuda.device_count()) if available else 0
    devices = [cuda.get_device_name(index) for index in range(count)]
    bf16 = bool(cuda.is_bf16_supported()) if available else False
    distributed = torch_module.distributed
    backends = {
        "nccl": bool(distributed.is_nccl_available()),
        "gloo": bool(distributed.is_gloo_available()),
    }
    return {
        "torch_version": torch_module.__version__,
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "cuda_available": available,
        "device_count": count,
        "devices": devices,
        "bf16_supported": bf16,
        "distributed_backends": backends,
    }


def require_cuda(summary: dict[str, Any]) -> None:
    if not summary["cuda_available"]:
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build, install a compatible "
            "NVIDIA driver, then rerun `make gpu-verify`; use `make train-smoke` for CPU-only validation."
        )
    if not summary["distributed_backends"].get("nccl", False):
        raise RuntimeError(
            "CUDA is available but NCCL is unavailable. Install a PyTorch build with NCCL support "
            "or run single-GPU verification without distributed training."
        )


def run_smoke(torch_module: Any) -> dict[str, Any]:
    """Exercise the real trainer for one deterministic CUDA step."""
    from ml.models import BaseModelConfig, KothaGPT, ModelConfig, TrainingConfig
    from ml.trainer import CausalLMDataset, train

    device = "cuda"
    config = BaseModelConfig(
        model=ModelConfig(
            vocab_size=64, hidden_size=32, num_layers=1, num_heads=2, max_position_embeddings=16
        ),
        training=TrainingConfig(batch_size=2, max_steps=1, mixed_precision="bf16"),
        data=type("DataConfig", (), {"tokenizer_path": "", "train": "", "validation": ""})(),
    )
    tokens = torch_module.randint(0, 64, (8, 16), device="cpu")
    dataset = CausalLMDataset(tokens)
    with tempfile.TemporaryDirectory(prefix="kothagpt-gpu-smoke-") as directory:
        result = train(
            KothaGPT(config.model), config, dataset, None, out_dir=Path(directory), device=device
        )
    return {
        "device": device,
        "step": result["step"],
        "tokens_per_sec": result.get("tokens_per_sec"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-smoke", action="store_true", help="only inspect hardware")
    args = parser.parse_args(argv)

    try:
        import torch

        summary = inspect_environment(torch)
        print(summary)
        require_cuda(summary)
        if not args.no_smoke:
            print({"smoke": run_smoke(torch)})
    except (ImportError, RuntimeError) as error:
        print(f"GPU verification failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

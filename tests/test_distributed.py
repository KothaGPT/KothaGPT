"""Distributed training tests (WS-7).

DDP needs a spawn context: forking after a previous ``train()`` run (e.g. a
reference run) deadlocks because inherited threads clash with the new process
group, and queues from the fork context can't be sent across spawn children.
We use ``mp.get_context("spawn")`` plus a queue created from that context.

Parity design: the dataset is built of N IDENTICAL blocks, so every block
produces the same loss. DistributedSampler (shuffle=False) splits indices
``[rank::2]``, so any subset mean equals the full-dataset mean exactly. The
reference single-process run and each DDP rank therefore report the same loss,
and every history line matches the reference.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from ml.models import KothaGPT
from ml.trainer import CausalLMDataset, init_distributed, train
from ml.trainer.distributed import DistributedConfig, destroy_distributed

CPU_COUNT = mp.cpu_count()


def _worker(config, blocks, out_dir, result_q, rank, port):
    dist = DistributedConfig(
        world_size=2,
        rank=rank,
        local_rank=rank,
        master_addr="127.0.0.1",
        master_port=port,
        backend="gloo",
        distributed=True,
    )
    dist = init_distributed(dist)
    torch.manual_seed(12345)
    model = KothaGPT(config.model)
    dataset = CausalLMDataset(blocks)
    result = train(
        model,
        config,
        dataset,
        None,
        out_dir=out_dir,
        device="cpu",
        dist=dist,
    )
    result_q.put(result)
    destroy_distributed()


@pytest.mark.skipif(CPU_COUNT < 2, reason="needs at least 2 CPUs")
def test_ddp_training_matches_single_process(config, tmp_path: Path) -> None:
    block_size = config.model.max_position_embeddings
    n_blocks = 32
    blocks = torch.tensor(
        [[i % block_size for i in range(block_size)] for _ in range(n_blocks)],
        dtype=torch.long,
    )
    dataset = CausalLMDataset(blocks)

    torch.manual_seed(12345)
    reference = train(
        KothaGPT(config.model),
        config,
        dataset,
        None,
        out_dir=tmp_path / "ref",
        device="cpu",
    )

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    port = "29500"
    procs = []
    for rank in (0, 1):
        p = ctx.Process(
            target=_worker,
            args=(config, blocks, tmp_path / "ddp", result_q, rank, port),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(120)
        assert p.exitcode == 0, "rank worker crashed"

    ddp_results = [result_q.get(timeout=10) for _ in range(2)]
    assert reference["step"] == 4
    assert all(r["step"] == 4 for r in ddp_results)
    assert all(abs(reference["loss"] - r["loss"]) < 1e-5 for r in ddp_results)

    history = (tmp_path / "ddp" / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    step_losses = [json.loads(line)["loss"] for line in history]
    assert len(step_losses) == 4
    assert all(abs(l - reference["loss"]) < 1e-5 for l in step_losses[-1:])
    assert all(s > reference["loss"] for s in step_losses[:-1])

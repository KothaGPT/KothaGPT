"""WS-2 Phase C: MoE inference engine + expert-parallel batching smoke tests."""

from __future__ import annotations

import pytest

from ml.inference.expert_parallel import ExpertParallelBatcher, expert_parallel_smoke
from ml.inference.moe_engine import MoEEngine, MoEEngineConfig, moe_engine_smoke
from ml.models.moe.router import Router


def test_expert_parallel_smoke_returns_counts():
    metrics = expert_parallel_smoke()
    assert metrics["experts_used"] >= 1
    # With top-2 routing each of the 8 tokens is assigned to 2 experts.
    assert metrics["tokens_routed"] == 16


def test_expert_parallel_batcher_groups_tokens_by_expert():
    torch = pytest.importorskip("torch")
    router = Router(hidden_size=16, num_experts=4, active_experts=2)
    batcher = ExpertParallelBatcher(router)
    hidden = torch.randn(8, 16)
    batches = batcher.batch(hidden, router)
    total = sum(len(b.token_indices) for b in batches)
    assert total >= 8
    for batch in batches:
        assert batch.hidden.shape[0] == len(batch.token_indices)


def test_expert_parallel_batcher_rejects_empty_input():
    torch = pytest.importorskip("torch")
    router = Router(hidden_size=16, num_experts=4, active_experts=2)
    batcher = ExpertParallelBatcher(router)
    with pytest.raises(ValueError, match="empty token batch"):
        batcher.batch(torch.zeros(0, 16), router)


def test_expert_parallel_batcher_rejects_wrong_hidden_dim():
    torch = pytest.importorskip("torch")
    router = Router(hidden_size=16, num_experts=4, active_experts=2)
    batcher = ExpertParallelBatcher(router)
    with pytest.raises(ValueError, match="final dimension 16"):
        batcher.batch(torch.randn(4, 8), router)


def test_moe_engine_smoke_runs_and_returns_stats():
    metrics = moe_engine_smoke()
    assert metrics["routing_table_size"] == 4
    assert metrics["steps_generated"] == 4
    assert metrics["experts"] == 4
    assert metrics["step_shape"] == (1, 1, 16)


def test_moe_engine_build_routing_table_covers_all_experts():
    engine = MoEEngine(MoEEngineConfig(num_experts=4, active_experts=2))
    table = engine.build_routing_table()
    assert set(table) == {0, 1, 2, 3}
    for bucket, experts in table.items():
        assert len(experts) == 2
        assert bucket in experts


def test_moe_engine_generate_yields_hidden_state_steps():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    engine = MoEEngine(MoEEngineConfig(max_new_tokens=3))
    initial = torch.randn(1, 4, 16)
    steps = list(engine.generate(initial, max_new_tokens=3))
    assert len(steps) == 3
    for step in steps:
        assert step.shape == (1, 1, 16)


def test_moe_engine_experts_property_returns_module_list():
    engine = MoEEngine(MoEEngineConfig(num_experts=4))
    assert len(engine.experts) == 4


def test_moe_engine_forward_rejects_wrong_shape():
    torch = pytest.importorskip("torch")
    engine = MoEEngine(MoEEngineConfig(hidden_size=16, num_heads=4))
    with pytest.raises(ValueError, match="hidden states of shape"):
        engine.forward(torch.randn(4, 16))


def test_moe_engine_config_validation():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        MoEEngineConfig(hidden_size=0)
    with pytest.raises(ValueError, match="num_experts must be positive"):
        MoEEngineConfig(num_experts=0)
    with pytest.raises(ValueError, match="active_experts must be between"):
        MoEEngineConfig(num_experts=4, active_experts=5)
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        MoEEngineConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MoEEngineConfig(hidden_size=16, num_heads=3)

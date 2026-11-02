from __future__ import annotations

from ml.inference.router import (
    ComplexitySignal,
    ModelRouter,
    RouteDecision,
    TaskClass,
)
from ml.inference.router_config import load_router_config


def test_router_config_file_exists_and_has_required_blocks():
    config = load_router_config()
    for block in ("tiers", "task_class_tiers", "escalation_chain", "tier_quality"):
        assert block in config, f"router config missing block: {block}"
    assert set(config["tiers"]) == {"nano", "small", "medium", "large"}


def test_router_config_task_class_tiers_cover_all_task_classes():
    config = load_router_config()
    tiers = config["task_class_tiers"]
    for task_class in TaskClass:
        assert task_class.value in tiers, f"missing task class {task_class.value}"


def test_router_config_escalation_chain_is_total_and_terminates():
    config = load_router_config()
    chain = config["escalation_chain"]
    assert chain["large"] == []
    for tier, next_tiers in chain.items():
        for next_tier in next_tiers:
            assert next_tier in chain, f"{tier} escalates to unknown {next_tier}"


def test_router_config_tier_quality_has_all_four_tiers():
    config = load_router_config()
    assert set(config["tier_quality"]) == {"nano", "small", "medium", "large"}


def test_model_router_uses_config_file_task_class_tiers():
    router = ModelRouter()
    # Use a low quality floor so the default tier is never auto-escalated;
    # this verifies the task-class → tier mapping from the config file.
    for task_class, expected in (
        (TaskClass.AUTOCOMPLETE, "nano"),
        (TaskClass.SHORT_CHAT, "nano"),
        (TaskClass.GENERAL_CHAT, "small"),
        (TaskClass.CODING_ASSIST, "small"),
        (TaskClass.RAG, "small"),
        (TaskClass.REASONING, "medium"),
        (TaskClass.LONG_CONTEXT, "medium"),
        (TaskClass.AGENT, "medium"),
        (TaskClass.MULTIMODAL, "large"),
        (TaskClass.FRONTIER, "large"),
    ):
        decision = router.route(
            ComplexitySignal(prompt_length=10, expected_tokens=10, task_class=task_class),
            quality_floor=0.0,
        )
        assert decision.tier == expected, f"{task_class.value} should route to {expected}"
        assert decision.model_id == f"kothagpt-{expected}"


def test_model_router_returns_route_decision_with_confidence():
    router = ModelRouter()
    decision = router.route(ComplexitySignal(prompt_length=10, expected_tokens=10, task_class=TaskClass.REASONING))
    assert isinstance(decision, RouteDecision)
    assert 0.0 <= decision.confidence <= 1.0


def test_model_router_escalation_chain_matches_config():
    router = ModelRouter()
    assert router.escalate("nano") == ["small", "medium", "large"]
    assert router.escalate("small") == ["medium", "large"]
    assert router.escalate("medium") == ["large"]
    assert router.escalate("large") == []


def test_model_router_should_escalate_uses_measured_quality():
    router = ModelRouter()
    # Nano measured quality 0.62 < 0.8 floor → escalate.
    assert router.should_escalate("nano", 0.8) is True
    # Large measured quality 0.96 >= 0.8 floor → no escalation.
    assert router.should_escalate("large", 0.8) is False
    # Unknown tier has no measured quality → no escalation.
    assert router.should_escalate("bogus", 0.8) is False


def test_model_router_escalates_when_quality_floor_not_met():
    router = ModelRouter()
    # With a floor above every tier's measured quality, routing should climb
    # the escalation chain until it reaches the last tier.
    signal = ComplexitySignal(prompt_length=10, expected_tokens=10, task_class=TaskClass.REASONING)
    # Default floor 0.8: medium (0.88) meets it, so no escalation.
    decision = router.route(signal)
    assert decision.tier == "medium"
    # Force an impossible floor so every tier escalates to large.
    router._tier_quality = {t: 0.0 for t in ("nano", "small", "medium", "large")}
    decision = router.route(signal, quality_floor=0.8)
    assert decision.tier == "large"
    assert decision.escalation_reason is not None


def test_model_router_accepts_explicit_tier_quality_override():
    router = ModelRouter(tier_quality={"nano": 0.95, "small": 0.95, "medium": 0.95, "large": 0.95})
    assert router.should_escalate("nano", 0.8) is False


def test_model_router_latency_sla_matches_config():
    router = ModelRouter()
    config = load_router_config()
    for tier, expected_ms in config["latency_sla_ms"].items():
        assert router.latency_sla_ms(tier) == expected_ms


def test_model_router_missing_config_file_falls_back_to_defaults(tmp_path):
    router = ModelRouter(config_path=str(tmp_path / "does-not-exist.yaml"))
    # Defaults: reasoning → medium.
    decision = router.route(
        ComplexitySignal(prompt_length=10, expected_tokens=10, task_class=TaskClass.REASONING)
    )
    assert decision.tier == "medium"
    assert router.escalate("nano") == ["small", "medium", "large"]
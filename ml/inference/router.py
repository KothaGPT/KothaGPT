"""Model routing — sends requests to the right model tier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

TIER_QUALITY: dict[str, float] = {
    "nano": 0.62,
    "small": 0.76,
    "medium": 0.88,
    "large": 0.96,
}


class TaskClass(str, Enum):
    AUTOCOMPLETE = "autocomplete"
    SHORT_CHAT = "short_chat"
    GENERAL_CHAT = "general_chat"
    CODING_ASSIST = "coding_assist"
    RAG = "rag"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    AGENT = "agent"
    MULTIMODAL = "multimodal"
    FRONTIER = "frontier"


@dataclass
class RouteDecision:
    tier: str
    model_id: str
    confidence: float
    escalation_reason: str | None = None


@dataclass
class ComplexitySignal:
    prompt_length: int
    expected_tokens: int
    task_class: TaskClass
    user_priority: str = "normal"
    latency_sla_ms: int = 5000


class ModelRouter:
    """Intelligent router sending requests to the right model tier."""

    def __init__(
        self,
        config_path: str | None = None,
        tier_quality: dict[str, float] | None = None,
    ):
        self.config_path = config_path or "ml/inference/router_config.yaml"
        self._tier_quality: dict[str, float] = {**TIER_QUALITY, **(tier_quality or {})}
        self._tier_thresholds: dict[TaskClass, str] = {
            TaskClass.AUTOCOMPLETE: "nano",
            TaskClass.SHORT_CHAT: "nano",
            TaskClass.GENERAL_CHAT: "small",
            TaskClass.CODING_ASSIST: "small",
            TaskClass.RAG: "small",
            TaskClass.REASONING: "medium",
            TaskClass.LONG_CONTEXT: "medium",
            TaskClass.AGENT: "medium",
            TaskClass.MULTIMODAL: "large",
            TaskClass.FRONTIER: "large",
        }
        self._escalation_chain: dict[str, list[str]] = {
            "nano": ["small", "medium", "large"],
            "small": ["medium", "large"],
            "medium": ["large"],
            "large": [],
        }

    def route(self, signal: ComplexitySignal) -> RouteDecision:
        """Classify request complexity and return tier assignment."""
        tier = self._tier_thresholds.get(signal.task_class, "small")
        return RouteDecision(
            tier=tier,
            model_id=f"kothagpt-{tier}",
            confidence=0.9,
        )

    def escalate(self, current_tier: str) -> list[str]:
        """Return escalation chain for a given tier."""
        return self._escalation_chain.get(current_tier, [])

    def should_escalate(self, current_tier: str, quality_floor: float) -> bool:
        """Return True when the tier's measured quality is below the floor."""
        measured_quality = self._tier_quality.get(current_tier)
        if measured_quality is None:
            return False
        return measured_quality < quality_floor

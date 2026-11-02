"""Model routing — sends requests to the right model tier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

TIER_QUALITY: dict[str, float] = {
    "nano": 0.62,
    "small": 0.76,
    "medium": 0.88,
    "large": 0.96,
}

_DEFAULT_CONFIG_PATH = "ml/inference/router_config.yaml"


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


def _load_config(config_path: str | None) -> dict:
    """Load the routing YAML, falling back to in-code defaults on failure."""
    import yaml

    path = Path(config_path or _DEFAULT_CONFIG_PATH)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _build_thresholds(config: dict) -> dict[TaskClass, str]:
    """Build the task-class → tier map from the YAML, falling back to defaults."""
    defaults: dict[TaskClass, str] = {
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
    raw = config.get("task_class_tiers")
    if not isinstance(raw, dict):
        return defaults
    result: dict[TaskClass, str] = {}
    for key, tier in raw.items():
        try:
            task_class = TaskClass(str(key).strip().lower())
        except ValueError:
            continue
        if not isinstance(tier, str) or not tier:
            continue
        result[task_class] = tier
    # Merge: explicit entries override defaults, unknown entries are dropped.
    merged = dict(defaults)
    merged.update(result)
    return merged


def _build_escalation_chain(config: dict) -> dict[str, list[str]]:
    """Build the escalation chain from the YAML, falling back to defaults."""
    defaults: dict[str, list[str]] = {
        "nano": ["small", "medium", "large"],
        "small": ["medium", "large"],
        "medium": ["large"],
        "large": [],
    }
    raw = config.get("escalation_chain")
    if not isinstance(raw, dict):
        return defaults
    result: dict[str, list[str]] = {}
    for tier, chain in raw.items():
        if not isinstance(tier, str) or not tier:
            continue
        if not isinstance(chain, list) or not all(isinstance(t, str) for t in chain):
            continue
        result[tier] = list(chain)
    merged = dict(defaults)
    merged.update(result)
    return merged


def _build_tier_quality(config: dict) -> dict[str, float]:
    """Build the measured-quality map from the YAML, falling back to defaults."""
    raw = config.get("tier_quality")
    if not isinstance(raw, dict):
        return dict(TIER_QUALITY)
    result: dict[str, float] = {}
    for tier, value in raw.items():
        if not isinstance(tier, str) or not tier:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result[tier] = float(value)
    merged = dict(TIER_QUALITY)
    merged.update(result)
    return merged


class ModelRouter:
    """Intelligent router sending requests to the right model tier."""

    def __init__(
        self,
        config_path: str | None = None,
        tier_quality: dict[str, float] | None = None,
    ):
        self.config_path = config_path or _DEFAULT_CONFIG_PATH
        config = _load_config(self.config_path)
        self._tier_quality: dict[str, float] = _build_tier_quality(config)
        if tier_quality:
            self._tier_quality.update({k: float(v) for k, v in tier_quality.items()})
        self._tier_thresholds: dict[TaskClass, str] = _build_thresholds(config)
        self._escalation_chain: dict[str, list[str]] = _build_escalation_chain(config)
        # Optional latency SLA targets per tier (used by circuit breaker).
        sla = config.get("latency_sla_ms") if isinstance(config, dict) else None
        self._latency_sla_ms: dict[str, int] = {}
        if isinstance(sla, dict):
            for tier, value in sla.items():
                if isinstance(tier, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._latency_sla_ms[tier] = int(value)

    def route(self, signal: ComplexitySignal, quality_floor: float = 0.8) -> RouteDecision:
        """Classify request complexity and return tier assignment.

        If the default tier for the task class is below ``quality_floor``,
        escalate through the chain until a tier meets the floor (or the chain
        is exhausted). The floor defaults to 0.8; callers can pass a lower
        value to allow lower-tier routing.
        """
        default_tier = self._tier_thresholds.get(signal.task_class, "small")
        tier = default_tier
        while self.should_escalate(tier, quality_floor):
            chain = self.escalate(tier)
            if not chain:
                break
            tier = chain[0]
        return RouteDecision(
            tier=tier,
            model_id=f"kothagpt-{tier}",
            confidence=0.9,
            escalation_reason=(
                "quality floor not met" if tier != default_tier else None
            ),
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

    def latency_sla_ms(self, tier: str) -> int | None:
        """Return the latency SLA for a tier, or None if unset."""
        return self._latency_sla_ms.get(tier)

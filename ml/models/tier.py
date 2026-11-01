from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class ModelTier(str, Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


TIER_CONFIGS: dict[ModelTier, str] = {
    ModelTier.NANO: "ml/configs/nano.yaml",
    ModelTier.SMALL: "ml/configs/small.yaml",
    ModelTier.MEDIUM: "ml/configs/medium.yaml",
    ModelTier.LARGE: "ml/configs/large.yaml",
}

TIER_METADATA: dict[ModelTier, dict[str, Any]] = {
    ModelTier.NANO: {
        "params_approx": "0.3B",
        "hidden_size": 256,
        "num_layers": 8,
        "context_window": 2048,
        "device_profile": "cpu",
        "quant_level": "int8",
        "cost_per_1k_tokens": 0.001,
        "target_latency_p95_ms": 100,
    },
    ModelTier.SMALL: {
        "params_approx": "1.5B",
        "hidden_size": 768,
        "num_layers": 12,
        "context_window": 4096,
        "device_profile": "cpu",
        "quant_level": "none",
        "cost_per_1k_tokens": 0.005,
        "target_latency_p95_ms": 300,
    },
    ModelTier.MEDIUM: {
        "params_approx": "7B",
        "hidden_size": 2048,
        "num_layers": 24,
        "context_window": 8192,
        "device_profile": "gpu",
        "quant_level": "int8",
        "cost_per_1k_tokens": 0.02,
        "target_latency_p95_ms": 1000,
    },
    ModelTier.LARGE: {
        "params_approx": "14B+",
        "hidden_size": 4096,
        "num_layers": 32,
        "context_window": 16384,
        "device_profile": "gpu",
        "quant_level": "int4",
        "cost_per_1k_tokens": 0.08,
        "target_latency_p95_ms": 2000,
    },
}


def get_tier_config(tier: ModelTier) -> str:
    return TIER_CONFIGS[tier]


def get_tier_metadata(tier: ModelTier) -> dict[str, Any]:
    return TIER_METADATA[tier]


def resolve_tier(tier: str | ModelTier) -> ModelTier:
    if isinstance(tier, ModelTier):
        return tier
    return ModelTier(tier)

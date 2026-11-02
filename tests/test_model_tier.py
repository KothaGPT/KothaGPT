from __future__ import annotations

from pathlib import Path

import pytest

from ml.models.config import BaseModelConfig, load_config
from ml.models.tier import (
    TIER_CONFIGS,
    TIER_METADATA,
    ModelTier,
    get_tier_config,
    get_tier_metadata,
    resolve_tier,
)

ROOT = Path(__file__).parents[1]


def _config_for(tier: ModelTier) -> BaseModelConfig:
    return load_config(ROOT / get_tier_config(tier))


def test_model_tier_enum_has_all_four_tiers():
    assert {t.value for t in ModelTier} == {"nano", "small", "medium", "large"}


def test_resolve_tier_accepts_string_and_enum():
    assert resolve_tier("nano") is ModelTier.NANO
    assert resolve_tier(ModelTier.SMALL) is ModelTier.SMALL
    with pytest.raises(ValueError):
        resolve_tier("bogus")


def test_tier_configs_point_to_existing_files():
    for tier, path in TIER_CONFIGS.items():
        assert (ROOT / path).is_file(), f"{tier} config missing: {path}"


def test_tier_metadata_matches_config_specs():
    # Nano
    nano = TIER_METADATA[ModelTier.NANO]
    assert nano["hidden_size"] == 256
    assert nano["context_window"] == 2048
    assert nano["device_profile"] == "cpu"
    assert nano["quant_level"] == "int8"
    # Small
    small = TIER_METADATA[ModelTier.SMALL]
    assert small["hidden_size"] == 768
    assert small["context_window"] == 4096
    # Medium
    medium = TIER_METADATA[ModelTier.MEDIUM]
    assert medium["hidden_size"] == 2048
    assert medium["context_window"] == 8192
    assert medium["device_profile"] == "gpu"
    # Large
    large = TIER_METADATA[ModelTier.LARGE]
    assert large["hidden_size"] == 4096
    assert large["context_window"] == 16384
    assert large["quant_level"] == "int4"


def test_tier_configs_validate_and_match_metadata():
    for tier in ModelTier:
        cfg = _config_for(tier)
        meta = TIER_METADATA[tier]
        assert cfg.model.hidden_size == meta["hidden_size"]
        assert cfg.model.max_position_embeddings == meta["context_window"]


def test_get_tier_config_and_metadata_round_trip():
    for tier in ModelTier:
        assert get_tier_config(tier) == TIER_CONFIGS[tier]
        assert get_tier_metadata(tier) == TIER_METADATA[tier]


def test_nano_config_has_minimal_compute():
    cfg = _config_for(ModelTier.NANO)
    assert cfg.model.num_layers == 8
    assert cfg.model.num_heads == 4
    assert cfg.training.batch_size == 4
    assert cfg.training.gradient_accumulation_steps == 8


def test_medium_config_has_extended_context_and_rope_scaling():
    cfg = _config_for(ModelTier.MEDIUM)
    assert cfg.model.num_layers == 24
    assert cfg.model.num_heads == 16
    assert cfg.model.rope_scaling == "linear"
    assert cfg.model.rope_scaling_factor == 2.0


def test_large_config_has_largest_dimensions():
    cfg = _config_for(ModelTier.LARGE)
    assert cfg.model.num_layers == 32
    assert cfg.model.num_heads == 32
    assert cfg.model.max_position_embeddings == 16384
    assert cfg.model.rope_scaling_factor == 4.0
    assert cfg.training.gradient_accumulation_steps == 32


def test_small_config_is_current_default():
    cfg = _config_for(ModelTier.SMALL)
    assert cfg.model.hidden_size == 768
    assert cfg.model.num_layers == 12
    assert cfg.model.num_heads == 12
    assert cfg.model.max_position_embeddings == 4096
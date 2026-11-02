from __future__ import annotations

from pathlib import Path

from ml.models.config import config_digest, load_config
from ml.models.tier import TIER_CONFIGS, ModelTier, get_tier_config

ROOT = Path(__file__).parents[1]


def _digest_for(tier: ModelTier) -> str:
    cfg = load_config(ROOT / get_tier_config(tier))
    return config_digest(cfg)


def test_each_tier_config_has_a_unique_digest():
    digests = {tier: _digest_for(tier) for tier in ModelTier}
    assert len(set(digests.values())) == len(ModelTier), "tiers must differ in digest"


def test_tier_config_digest_is_stable_across_loads():
    digest_a = _digest_for(ModelTier.SMALL)
    digest_b = _digest_for(ModelTier.SMALL)
    assert digest_a == digest_b
    assert len(digest_a) == 64  # sha256 hex


def test_tier_config_digest_changes_when_training_knobs_change(tmp_path: Path):
    import yaml

    src = ROOT / TIER_CONFIGS[ModelTier.SMALL]
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["training"]["max_steps"] = 999
    out = tmp_path / "small-modified.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    original = _digest_for(ModelTier.SMALL)
    modified = config_digest(load_config(out))
    assert original != modified


def test_all_tier_configs_resolve_to_known_model_tiers():
    for tier in ModelTier:
        assert TIER_CONFIGS[tier].startswith("ml/configs/")


def test_tier_config_paths_are_under_ml_configs():
    for tier, path in TIER_CONFIGS.items():
        assert path.startswith("ml/configs/")
        assert path.endswith(".yaml")
        assert tier.value in path
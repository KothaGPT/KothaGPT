"""WS-1/WS-6: ModelRegistry tier defaults carry artifact_path and content-addressable digest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ml.inference.registry import ModelRegistry, get_registry


def _file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_registry_is_singleton():
    assert get_registry() is get_registry()


def test_all_four_tiers_registered_with_ready_status():
    reg = ModelRegistry()
    for tier_id in ("kothagpt-nano", "kothagpt-small", "kothagpt-medium",
                    "kothagpt-large"):
        rec = reg.get(tier_id)
        assert rec is not None, f"{tier_id} not registered"
        assert rec.status == "ready", f"{tier_id} status is {rec.status!r}"


def test_tier_records_have_artifact_path_and_digest():
    reg = ModelRegistry()
    expected = {
        "kothagpt-nano": "ml/configs/nano.yaml",
        "kothagpt-small": "ml/configs/small.yaml",
        "kothagpt-medium": "ml/configs/medium.yaml",
        "kothagpt-large": "ml/configs/large.yaml",
    }
    for tier_id, cfg_file in expected.items():
        rec = reg.get(tier_id)
        assert rec is not None, f"{tier_id} not registered"
        assert rec.artifact_path == cfg_file, (
            f"{tier_id} artifact_path is {rec.artifact_path!r}, expected {cfg_file!r}"
        )
        assert rec.digest == _file_digest(cfg_file), (
            f"{tier_id} digest does not match file hash"
        )
        assert len(rec.digest) == 64, f"{tier_id} digest is not a sha256 hex string"


def test_digest_is_deterministic_across_instances():
    first = ModelRegistry().get("kothagpt-small").digest
    second = ModelRegistry().get("kothagpt-small").digest
    assert first == second
    assert first == _file_digest("ml/configs/small.yaml")


def test_non_tier_records_do_not_require_artifact_path():
    reg = ModelRegistry()
    base = reg.get("kothagpt")
    assert base is not None and base.status == "ready"
    moe = reg.get("kothagpt-moe")
    assert moe is not None and moe.status == "research"


def test_list_models_includes_all_default_records():
    reg = ModelRegistry()
    ids = {r.id for r in reg.list_models()}
    for tier_id in ("kothagpt-nano", "kothagpt-small", "kothagpt-medium",
                    "kothagpt-large", "kothagpt", "kothagpt-embed",
                    "kothagpt-rerank", "kothagpt-moe"):
        assert tier_id in ids, f"{tier_id} missing from registry list"

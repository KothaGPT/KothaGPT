"""Tests for the dataset provenance and license registry (KGPT-003)."""

from __future__ import annotations

import json
from pathlib import Path

from data.manifest import (
    DatasetRegistry,
    DatasetSource,
    LicenseAllowlist,
    load_registry,
    save_registry,
)


def _source(name: str = "wiki", license_id: str = "cc-by-sa-4.0") -> DatasetSource:
    return DatasetSource(
        name=name,
        url=f"https://example.com/{name}",
        license=license_id,
        provenance="test fixture",
    )


def test_source_requires_name_and_license():
    import pytest

    with pytest.raises(ValueError):
        DatasetSource(name="", url="u", license="cc-by-4.0")
    with pytest.raises(ValueError):
        DatasetSource(name="x", url="u", license="")


def test_registry_register_and_lookup():
    reg = DatasetRegistry()
    src = _source()
    reg.register(src)
    assert reg.get("wiki") is src
    assert reg.list() == [src]


def test_registry_rejects_duplicate_registration():
    import pytest

    reg = DatasetRegistry()
    reg.register(_source())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_source())


def test_registry_is_allowed_respects_allowlist():
    reg = DatasetRegistry()
    reg.register(_source(license_id="cc-by-sa-4.0"))
    ok, _ = reg.is_allowed("wiki")
    assert ok is True
    reg.register(_source("blocked", license_id="all-rights-reserved"))
    ok, reason = reg.is_allowed("blocked")
    assert ok is False
    assert "not in the allowlist" in reason


def test_registry_is_allowed_rejects_unregistered_source():
    reg = DatasetRegistry()
    ok, reason = reg.is_allowed("nope")
    assert ok is False
    assert "not registered" in reason


def test_registry_enforce_raises_on_blocked_source():
    import pytest

    reg = DatasetRegistry()
    reg.register(_source("bad", license_id="all-rights-reserved"))
    with pytest.raises(ValueError, match="dataset source rejected"):
        reg.enforce(["bad"])


def test_registry_enforce_passes_for_allowed_sources():
    reg = DatasetRegistry()
    reg.register(_source("good"))
    reg.enforce(["good"])  # should not raise


def test_allowlist_default_includes_common_open_licenses():
    allowlist = LicenseAllowlist()
    assert allowlist.is_allowed("cc-by-4.0")
    assert allowlist.is_allowed("mit")
    assert not allowlist.is_allowed("all-rights-reserved")
    assert not allowlist.is_allowed("")


def test_allowlist_custom_blocked_overrides_allowlist():
    allowlist = LicenseAllowlist(
        allowed=frozenset({"cc-by-4.0"}),
        blocked=frozenset({"cc-by-4.0"}),
    )
    assert allowlist.is_allowed("cc-by-4.0") is False


def test_registry_save_and_load_round_trip(tmp_path: Path):
    reg = DatasetRegistry()
    reg.register(_source("wiki"))
    path = save_registry(reg, tmp_path / "registry.json")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert any(s["name"] == "wiki" for s in data["sources"])

    loaded = load_registry(path)
    assert loaded.get("wiki") is not None
    assert loaded.get("wiki").license == "cc-by-sa-4.0"


def test_registry_can_be_constructed_from_missing_path():
    reg = DatasetRegistry(path="/nonexistent/path/registry.json")
    assert reg.list() == []


def test_source_digest_is_stable():
    src = _source()
    first = src.digest()
    second = _source().digest()
    assert first == second
    assert len(first) == 16
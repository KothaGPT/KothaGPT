"""Tests for the Phase 1A copyright/licensing gate."""

from __future__ import annotations

from data.pipeline import copyright
from data.pipeline.config import PipelineConfig, run_pipeline


def test_normalize_license_variants():
    cases = {
        "CC-BY-SA-4.0": "cc-by-sa-4.0",
        "Creative Commons Attribution ShareAlike 4.0": "cc-by-sa-4.0",
        "MIT": "mit",
        "Apache 2.0": "apache-2.0",
        "BSD 3-Clause": "bsd-3-clause",
        "Public Domain": "public-domain",
        "CC0": "cc0",
        "unlicense": "unlicense",
    }
    for raw, expected in cases.items():
        assert copyright.normalize_license(raw) == expected, raw


def test_is_allowed():
    assert copyright.is_allowed("CC-BY-SA-4.0")
    assert copyright.is_allowed("MIT")
    assert not copyright.is_allowed("Copyright All Rights Reserved")
    assert not copyright.is_allowed("CC-BY-NC-4.0")  # NC excluded by default


def test_is_allowed_custom_allowlist():
    assert copyright.is_allowed("CC-BY-NC-4.0", allowlist=frozenset({"cc-by-nc-4.0"}))
    assert not copyright.is_allowed("MIT", allowlist=frozenset({"cc0"}))


def test_resolve_license_from_map():
    mapping = {"bn_wiki/*": "CC-BY-SA-4.0", "raw.jsonl": "MIT", "*": "cc0"}
    assert copyright.resolve_license({"source": "bn_wiki/dump.txt"}, mapping) == "CC-BY-SA-4.0"
    assert copyright.resolve_license({"source": "raw.jsonl"}, mapping) == "MIT"
    assert copyright.resolve_license({"source": "other/thing.txt"}, mapping) == "cc0"
    assert copyright.resolve_license({"source": "x.txt"}, {"bn_wiki/*": "cc-by"}) is None
    assert copyright.resolve_license({"license": "Apache-2.0"}, {}) == "Apache-2.0"


def test_copyright_gate_missing_license_rejected_when_required():
    keep, reasons = copyright.copyright_gate(
        {"text": "বাংলা ভাষা", "source": "unknown.txt"},
        require_license=True,
    )
    assert not keep
    assert any("license" in r for r in reasons)


def test_copyright_gate_allowed_license():
    keep, reasons = copyright.copyright_gate(
        {"text": "বাংলা ভাষা", "source": "wiki.txt", "license": "CC-BY-SA-4.0"},
        require_license=True,
    )
    assert keep and not reasons
    # license resolved from map is attached to the record.
    keep, reasons = copyright.copyright_gate(
        {"text": "বাংলা ভাষা", "source": "wiki/a.txt"},
        license_map={"wiki/*": "CC-BY-SA-4.0"},
        require_license=True,
    )
    assert keep and not reasons
    assert "license" in {"text": "x", "source": "wiki/a.txt", "license": "CC-BY-SA-4.0"}


def test_copyright_gate_disallowed_license():
    keep, reasons = copyright.copyright_gate(
        {"text": "বাংলা ভাষা", "license": "All Rights Reserved"},
        require_license=True,
    )
    assert not keep
    assert any("not allowed" in r for r in reasons)


def test_known_copyrighted_title():
    blocklist = copyright.load_title_blocklist(None)
    assert blocklist == frozenset()
    assert not copyright.known_copyrighted_title("সাধারণ বই", blocklist)


def test_title_blocklist_gate(tmp_path):
    block_file = tmp_path / "titles.txt"
    block_file.write_text("গণিতের মজা\n", encoding="utf-8")
    blocklist = copyright.load_title_blocklist(block_file)
    keep, reasons = copyright.copyright_gate(
        {"text": "বাংলা ভাষা", "license": "cc0", "title": "গণিতের মজা"},
        require_license=True,
        title_blocklist=blocklist,
    )
    assert not keep
    assert any("known copyrighted" in r for r in reasons)


def test_full_pipeline_with_license_gate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 40, encoding="utf-8")
    (raw / "bad.txt").write_text("গোপন নথি একটি সংবেদনশীল পাঠ্য। " * 40, encoding="utf-8")
    license_map = tmp_path / "licenses.json"
    license_map.write_text('{"ok.txt": "CC-BY-SA-4.0"}\n', encoding="utf-8")

    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        require_license=True,
        license_map_path=str(license_map),
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)

    assert summary["raw"] == 2
    assert summary["after_license"] == 1  # bad.txt dropped for missing license
    assert summary["after_filter"] == 1
    assert summary["train"] + summary["validation"] == 1


def test_full_pipeline_without_license_gate_keeps_all(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 40, encoding="utf-8")
    (raw / "no_license.txt").write_text("গোপন নথি একটি সংবেদনশীল পাঠ্য। " * 40, encoding="utf-8")

    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        require_license=False,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["raw"] == 2
    assert summary["after_license"] == 2

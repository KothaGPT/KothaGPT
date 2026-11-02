"""Tests for the Phase 1A toxic-content filter."""

from __future__ import annotations

from data.pipeline import toxic
from data.pipeline.config import PipelineConfig, run_pipeline


def test_clean_text_passes():
    text = (
        "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গের মানুষের মাতৃভাষা। "
        "এই ভাষায় সাহিত্য, কবিতা ও গানের একটি দীর্ঘ ঐতিহ্য রয়েছে।"
    )
    keep, reasons = toxic.toxic_gate(text)
    assert keep and not reasons


def test_english_slur_blocked():
    keep, reasons = toxic.toxic_gate("i do not like that nigger guy")
    assert not keep
    assert any(r.startswith("toxic:") for r in reasons)


def test_bangla_slur_blocked():
    keep, reasons = toxic.toxic_gate("সে একজন বেশ্যা মানুষ।")
    assert not keep
    assert any("বেশ্যা" in r for r in reasons)


def test_word_boundary_matching():
    # "fucker" inside "motherfucker" is a hit; "fuck" alone must NOT match
    # inside a larger unrelated word.
    assert toxic.toxic_gate("fucking hell")[0] is False
    assert toxic.toxic_gate("fucke")[0] is True  # not a real word boundary hit
    assert toxic.toxic_gate("ফুকু")[0] is True  # similar-looking Bangla word


def test_case_insensitive():
    assert toxic.toxic_gate("What a BItch move")[0] is False


def test_optional_classifier():
    always_toxic = lambda text: 0.9
    always_clean = lambda text: 0.1

    keep, reasons = toxic.toxic_gate("কিছু লেখা", classifier=always_toxic)
    assert not keep
    assert any(r.startswith("toxic_classifier") for r in reasons)

    keep, reasons = toxic.toxic_gate("কিছু লেখা", classifier=always_clean)
    assert keep and not reasons


def test_classifier_threshold_tuning():
    borderline = lambda text: 0.7
    assert toxic.toxic_gate("কিছু লেখা", classifier=borderline, classifier_threshold=0.8)[0] is True
    assert toxic.toxic_gate("কিছু লেখা", classifier=borderline, classifier_threshold=0.6)[0] is False


def test_load_classifier(tmp_path):
    mod = tmp_path / "mytox.py"
    mod.write_text("def tox(text):\n    return 1.0 if 'খারাপ' in text else 0.0\n", encoding="utf-8")
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        fn = toxic.load_classifier("mytox:tox")
        assert fn("খারাপ") == 1.0
        assert fn("ভাল") == 0.0
    finally:
        sys.path.pop(0)

    try:
        toxic.load_classifier("mytox:missing")
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for missing callable")

    try:
        toxic.load_classifier("no-colon")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for malformed path")


def test_full_pipeline_drops_toxic(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text(
        "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গের মানুষের মাতৃভাষা। "
        "এই ভাষায় সাহিত্য, কবিতা ও গানের একটি দীর্ঘ ঐতিহ্য রয়েছে।",
        encoding="utf-8",
    )
    (raw / "toxic.txt").write_text("আজকের আলোচনা হলো একটা বেশ্যা লোকের গল্প।", encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        check_toxic=True,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["raw"] == 2
    assert summary["after_toxic"] == 1
    assert summary["after_filter"] == 2


def test_full_pipeline_no_toxic_check_keeps_all(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 5, encoding="utf-8")
    (raw / "toxic.txt").write_text("আজকের আলোচনা হলো একটা বেশ্যা লোকের গল্প।", encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        check_toxic=False,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["after_toxic"] == 2

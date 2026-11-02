"""Tests for the Phase 1A spam filter."""

from __future__ import annotations

from data.pipeline import spam
from data.pipeline.config import PipelineConfig, run_pipeline


def test_clean_text_passes():
    text = (
        "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গের মানুষের মাতৃভাষা। "
        "এই ভাষায় সাহিত্য, কবিতা ও গানের একটি দীর্ঘ ঐতিহ্য রয়েছে। "
        "প্রতিদিন লক্ষ লক্ষ মানুষ বাংলায় কথা বলে এবং লেখে।"
    )
    keep, reasons = spam.spam_gate(text)
    assert keep and not reasons


def test_spam_domain_definitive():
    keep, reasons = spam.spam_gate("বাংলা ভাষা নিয়ে কথা বলি। দেখুন https://1xbet.com/bonus")
    assert not keep
    assert any("spam_domain" in r for r in reasons)


def test_promo_phrases_bangla_and_english():
    keep, reasons = spam.spam_gate("ফ্রি রিচার্জ! পুরস্কার জিতুন! এখানে ক্লিক করুন")
    assert not keep
    assert any("spam_phrase" in r for r in reasons)
    assert spam.spam_gate("buy now cash prize free gift click here")[0] is False


def test_repetition_burst_and_caps():
    score, reasons = spam.spam_score("রিচার্জ " * 15 + "!!!!!!!!!!")
    assert score >= 0.6
    assert any("repetition" in r for r in reasons)
    assert spam.spam_gate("পুরস্কার " * 20)[0] is False  # pure repetition


def test_legit_news_not_flagged():
    text = (
        "জাতীয় পুরস্কার ২০২৪ ঘোষণা করা হয়েছে। দেশের শিক্ষার্থীরা এবার ভালো ফল করেছে। "
        "বিজ্ঞান মেলায় অনেক নতুন প্রকল্প প্রদর্শিত হয়।"
    )
    keep, _ = spam.spam_gate(text)
    assert keep


def test_threshold_tuning():
    text = "পুরস্কার জিতুন! ফ্রি রিচার্জ!"
    assert spam.spam_gate(text, threshold=0.6)[0] is False
    assert spam.spam_gate(text, threshold=0.99)[0] is True  # lenient threshold


def test_full_pipeline_drops_spam(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text(
        "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গের মানুষের মাতৃভাষা। "
        "এই ভাষায় সাহিত্য, কবিতা ও গানের একটি দীর্ঘ ঐতিহ্য রয়েছে। "
        "প্রতিদিন লক্ষ লক্ষ মানুষ বাংলায় কথা বলে এবং লেখে।" * 3,
        encoding="utf-8",
    )
    (raw / "spam.txt").write_text("পুরস্কার জিতুন! ফ্রি রিচার্জ! এখানে ক্লিক করুন! " * 10, encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        check_spam=True,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["raw"] == 2
    assert summary["after_spam"] == 1  # spam.txt dropped
    assert summary["after_filter"] == 2


def test_full_pipeline_no_spam_check_keeps_all(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 20, encoding="utf-8")
    (raw / "spam.txt").write_text("পুরস্কার জিতুন! ফ্রি রিচার্জ! এখানে ক্লিক করুন! " * 10, encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        check_spam=False,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["raw"] == 2
    assert summary["after_spam"] == 2

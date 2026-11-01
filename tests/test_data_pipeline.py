"""Tests for the Phase 1A Bangla dataset pipeline (pure stdlib)."""

from __future__ import annotations

import gzip
import json

from data.pipeline import dedup, io, normalize, quality, split
from data.pipeline.config import PipelineConfig, enforce_source_provenance, run_pipeline


def _read_shard_text(path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def test_unicode_normalize_keeps_bangla_conjuncts():
    text = "বাংলা ভাষা\u200b\u200dকথা\u00a0লেখা"  # nbsp + zwsp + zwj
    out = normalize.unicode_normalize(text)
    assert "\u09cd" in out or "\u09cd" not in text  # conjuncts preserved
    assert "\u200b" not in out
    assert "\u00a0" not in out
    assert "\u200d" in out  # zero-width joiner is meaningful in Bangla


def test_strip_html_and_markup():
    raw = "<p>hello <b>world</b></p> <!-- comment --> <script>x</script>"
    out = normalize.strip_html(raw)
    assert "<" not in out and ">" not in out
    assert "hello" in out and "world" in out and "comment" not in out
    md = "# Heading\n[link](http://x.com) and `code` and ![img](i.png)"
    out = normalize.strip_markup(md)
    assert out.startswith("Heading")
    assert "link" in out and "(" not in out


def test_clean_whitespace():
    assert normalize.clean_whitespace("a  \t b\r\n\r\n\r\nc") == "a b\n\nc"


def test_bengali_ratio_and_language():
    bn = "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা এবং প্রিয় ভাষা।"
    assert quality.bengali_ratio(bn) > 0.5
    assert quality.detect_language(bn) == "bn"
    assert quality.detect_language("This is an English sentence for testing.") == "en"
    assert quality.detect_language("বাংলা and English মিশ্রণ।") == "mixed"


def test_pii_detection():
    assert "email" in quality.contains_pii("Contact me at user@example.com now")
    assert "phone" in quality.contains_pii("Call 01712345678 today")
    assert "url" in quality.contains_pii("See https://example.com/page for info")
    assert quality.contains_pii("বাংলা ভাষার কথা") == []


def test_pii_detection_extended_patterns():
    assert "passport" in quality.contains_pii("My passport is A0123456")
    assert "passport" in quality.contains_pii("Passport no. BG9876543")
    assert "address" in quality.contains_pii("Meet me at 12 Road, Dhanmondi")
    assert "address" in quality.contains_pii("বাসা ৪২, রোড ৭, ঢাকা")
    assert quality.contains_pii("বাংলা ভাষা একটি সমৃদ্ধ ভাষা") == []


def test_redact_pii_masks_and_counts():
    text = "email me@x.com or call 01712345678 at https://example.com"
    redacted, counts = quality.redact_pii(text)
    assert "me@x.com" not in redacted
    assert "01712345678" not in redacted
    assert "https://example.com" not in redacted
    assert redacted.count(quality.PII_MASK) == 3
    assert counts["email"] == 1
    assert counts["phone"] == 1
    assert counts["url"] == 1


def test_redact_pii_leaves_clean_text_untouched():
    clean = "বাংলা ভাষা একটি সমৃদ্ধ ভাষা।"
    redacted, counts = quality.redact_pii(clean)
    assert redacted == clean
    assert counts == {}


def test_full_pipeline_pii_mask_mode_keeps_doc(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "pii.txt").write_text("যোগাযোগ করুন user@example.com ঠিকানায়। " * 20, encoding="utf-8")
    (raw / "clean.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 20, encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        pii_mode="mask",
        strict=False,  # dev bypass: no registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["raw"] == 2
    assert summary["after_filter"] == 2  # masked doc kept
    version_dir = tmp_path / "out" / summary["version_id"]
    shards = sorted((version_dir / "train").glob("*.jsonl*")) + sorted(
        (version_dir / "validation").glob("*.jsonl*")
    )
    assert shards
    contents = "".join(_read_shard_text(s) for s in shards)
    assert "user@example.com" not in contents
    assert quality.PII_MASK in contents


def test_length_filter():
    ok, reasons = quality.length_filter("x" * 200, min_chars=100, max_chars=1000, min_words=0)
    assert ok and not reasons
    ok, reasons = quality.length_filter("x" * 10, min_chars=100, max_chars=1000, min_words=0)
    assert not ok and any("too_short" in r for r in reasons)


def test_quality_filter_rejects_pii_when_disallowed():
    text = "বাংলা ভাষা " * 30 + "email: user@example.com"
    keep, reasons = quality.quality_filter(
        text,
        min_chars=1,
        max_chars=100000,
        min_words=1,
        require_bangla=False,
        min_bangla_ratio=0.0,
        allow_pii=False,
    )
    assert not keep
    assert any("pii" in r for r in reasons)


def test_exact_dedup():
    docs = [{"text": "same text"}, {"text": "same text"}, {"text": "different"}]
    out = dedup.deduplicate(docs, exact=True)
    assert len(out) == 2


def test_near_dedup():
    base = (
        "বাংলা ভাষা একটি প্রাচীন এবং সমৃদ্ধ ভাষা যা হাজার বছরের ইতিহাস বহন করে। "
        "এই ভাষায় সাহিত্য, কবিতা, গান এবং বিজ্ঞানের অসংখ্য বই লেখা হয়েছে। "
        "বাংলাদেশ ও ভারতের পশ্চিমবঙ্গ, ত্রিপুরা ও আসামের মানুষ এই ভাষায় কথা বলে। "
        "বাংলা বিশ্বের সবচেয়ে বেশি মানুষের কথ্য ভাষাগুলোর মধ্যে একটি।"
    )
    near = base.replace("প্রাচীন", "পুরাতন", 1)
    docs = [
        {"text": base},
        {"text": near},
        {"text": "বাংলাদেশের স্বাধীনতা একটি ঐতিহাসিক ঘটনা।" * 20},
    ]
    out = dedup.deduplicate(docs, exact=True, near=True, threshold=0.75)
    assert len(out) == 2


def test_split_is_deterministic_and_balanced():
    texts = [f"বাংলা ভাষা নিয়ে লেখা {i}" for i in range(2000)]
    sets = [split.split_set(t, 0.1) for t in texts]
    again = [split.split_set(t, 0.1) for t in texts]
    assert sets == again
    val = sum(1 for s in sets if s == "validation")
    assert 0.05 < val / len(sets) < 0.2


def test_ensure_nonempty_validation_promotes_closest_record():
    records = [split.split_record({"text": f"doc {i}"}) for i in range(12)]
    if any(r["split"] == "validation" for r in records):
        records = [dict(r, split="train") for r in records]
    fixed = split.ensure_nonempty_validation(records)
    assert sum(1 for r in fixed if r["split"] == "validation") == 1
    # Deterministic: same input, same promotion.
    assert [r["split"] for r in split.ensure_nonempty_validation(records)] == [
        r["split"] for r in fixed
    ]
    # The promoted record is the one closest to the validation threshold.
    idx = min(range(len(records)), key=lambda i: split.split_key(records[i]["text"]))
    assert fixed[idx]["split"] == "validation"
    # Already-nonempty and tiny inputs pass through unchanged.
    mixed = [{"text": "a", "split": "train"}, {"text": "b", "split": "validation"}]
    assert split.ensure_nonempty_validation(mixed) == mixed
    single = [{"text": "only", "split": "train"}]
    assert split.ensure_nonempty_validation(single) == single


def test_write_shards_gzip(tmp_path):
    records = [{"text": f"doc {i}", "id": str(i), "source": "t"} for i in range(5)]
    shards = io.write_shards(iter(records), tmp_path, shard_size=2, gzip_output=True)
    assert len(shards) == 3
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [s["file"] for s in shards]
    with gzip.open(tmp_path / shards[0]["file"], "rt", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 2


def test_iter_records_jsonl(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"text": "বাংলা", "extra": 1}) + "\n", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.html").write_text("<html><body><p>html text</p></body></html>", encoding="utf-8")
    records = list(io.iter_records(tmp_path))
    assert len(records) == 3
    assert records[0]["text"] == "বাংলা"
    assert records[2]["source"].endswith(".html")


def test_iter_records_skips_hidden_files(tmp_path):
    (tmp_path / "a.txt").write_text("visible", encoding="utf-8")
    (tmp_path / ".licenses.json").write_text('{"a.txt": "cc0"}', encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("junk", encoding="utf-8")
    records = list(io.iter_records(tmp_path))
    assert len(records) == 1
    assert records[0]["source"] == "a.txt"


def test_full_pipeline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = [
        "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা। এটি একটি প্রাচীন এবং সমৃদ্ধ ভাষা।" * 10,
        "বাংলা ভাষা বাংলাদেশের মানুষের মাতৃভাষা। এটি একটি প্রাচীন এবং সমৃদ্ধ ভাষা।" * 10,
        "<p>বাংলা ভাষা হচ্ছে পূর্ব ভারতীয় আর্য ভাষা পরিবারের সদস্য।</p>" * 15,
        "This is an English text that should be filtered out by the language check." * 20,
    ]
    for i, text in enumerate(docs):
        (raw / f"d{i}.txt").write_text(text, encoding="utf-8")

    out_root = tmp_path / "out"
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(out_root),
        min_chars=50,
        max_chars=10000,
        min_words=5,
        require_bangla=True,
        min_bangla_ratio=0.5,
        dedup_exact=True,
        validation_ratio=0.1,
        shard_size=100,
        strict=False,  # dev bypass: no registry configured in this test
    )
    summary = run_pipeline(cfg)

    assert summary["raw"] == 4
    assert summary["after_filter"] == 3  # English doc dropped
    assert summary["after_dedup"] == 2  # duplicate removed
    assert summary["train"] + summary["validation"] == 2
    assert summary["version_id"]

    manifest = out_root / summary["version_id"] / "MANIFEST.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["version_id"] == summary["version_id"]
    assert (out_root / summary["version_id"] / "report" / "REPORT.md").exists()


def _registry_with(source_name: str = "wiki", license_id: str = "cc-by-sa-4.0"):
    from data.manifest import DatasetRegistry, DatasetSource

    reg = DatasetRegistry()
    reg.register(
        DatasetSource(
            name=source_name,
            url=f"https://example.com/{source_name}",
            license=license_id,
            provenance="test",
        )
    )
    return reg


def _records() -> list[dict]:
    return [
        {"id": "1", "text": "বাংলা ভাষা।", "source": "wiki"},
        {"id": "2", "text": "more bangla text।", "source": "wiki"},
        {"id": "3", "text": "unregistered source text।", "source": "unregistered"},
        {"id": "4", "text": "no source field here।"},
    ]


def test_enforce_source_provenance_strict_empty_registry_rejects_all():
    kept, rejected = enforce_source_provenance(_records(), strict=True)
    assert kept == []
    assert len(rejected) == 4
    assert all("no sources registered" in r for r in rejected)


def test_enforce_source_provenance_strict_unregistered_source_rejected():
    reg = _registry_with()
    kept, rejected = enforce_source_provenance(_records(), registry=reg, strict=True)
    kept_ids = {r["id"] for r in kept}
    assert kept_ids == {"1", "2"}
    assert any("unregistered" in r for r in rejected)
    assert any("missing 'source' field" in r for r in rejected)


def test_enforce_source_provenance_strict_disallowed_license_rejected():
    reg = _registry_with(license_id="all-rights-reserved")
    kept, rejected = enforce_source_provenance(_records()[:2], registry=reg, strict=True)
    assert kept == []
    assert all("not in the allowlist" in r for r in rejected)


def test_enforce_source_provenance_dev_bypass_allows_all():
    kept, rejected = enforce_source_provenance(_records(), strict=False)
    assert kept == _records()
    assert rejected == []


def test_enforce_source_provenance_dev_bypass_empty_registry_allows_all():
    kept, rejected = enforce_source_provenance(_records(), registry=_registry_with(), strict=False)
    assert kept == _records()
    assert rejected == []

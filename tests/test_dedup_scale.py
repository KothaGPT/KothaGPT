"""Tests for B1 scale dedup: Bloom filter, cross-source state, sharded LSH."""

from __future__ import annotations

from data.pipeline import dedup
from data.pipeline.config import PipelineConfig, run_pipeline


def test_bloom_filter_membership():
    bf = dedup.BloomFilter(capacity=1000, fp_rate=0.01)
    keys = [f"key-{i}" for i in range(100)]
    for k in keys:
        assert k not in bf
        bf.add(k)
    for k in keys:
        assert k in bf
    assert bf.count == 100
    assert bf.size_bytes > 0


def test_bloom_filter_false_positive_rate_is_bounded():
    # Empirically the FP rate should stay close to the configured value.
    bf = dedup.BloomFilter(capacity=500, fp_rate=0.01)
    for i in range(500):
        bf.add(f"in-{i}")
    false_positives = sum(1 for i in range(500) if f"out-{i}" in bf)
    assert false_positives / 500 < 0.1


def test_bloom_filter_validation():
    try:
        dedup.BloomFilter(capacity=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero capacity")
    try:
        dedup.BloomFilter(capacity=100, fp_rate=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for fp_rate 0.0")


def test_bloom_filter_as_exact_store():
    docs = [{"text": "same"}, {"text": "same"}, {"text": "other"}]
    out, counts = dedup.deduplicate_with_stats(
        docs, exact=True, exact_store=dedup.BloomFilter(100, 0.001)
    )
    assert len(out) == 2
    assert counts["removed_exact"] == 1


def test_exact_dedup_state_persists_across_runs(tmp_path):
    state = tmp_path / "dedup-state.txt"
    first = dedup.ExactDedupState(state)
    assert first.count == 0
    assert first.is_new("hello world")
    first.mark("hello world")
    first.save()
    assert state.exists()

    second = dedup.ExactDedupState(state)
    assert second.count == 1
    assert not second.is_new("hello world")
    assert second.is_new("other text")


def test_exact_dedup_state_round_trip_text(tmp_path):
    state = tmp_path / "state.txt"
    store = dedup.ExactDedupState(state)
    store.mark("বাংলা ভাষা")
    store.save()

    docs = [{"text": "বাংলা ভাষা"}, {"text": "আরেকটা লেখা"}]
    out, counts = dedup.deduplicate_with_stats(
        docs, exact=True, exact_store=dedup.ExactDedupState(state)
    )
    assert len(out) == 1  # cross-source: first doc already in state
    assert counts["removed_exact"] == 1


def test_near_duplicate_groups_sharded_detects_edits():
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
    groups = dedup.near_duplicate_groups_sharded(docs, threshold=0.75)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_near_sharded_matches_in_memory_behavior():
    base = "একটি প্রাচীন ভাষার ইতিহাস অনেক দীর্ঘ। " * 30
    near = base.replace("প্রাচীন", "পুরাতন", 1)
    docs = [{"text": base}, {"text": near}, {"text": "সম্পূর্ণ ভিন্ন লেখা। " * 30}]

    in_mem = dedup.deduplicate(docs, exact=True, near=True, threshold=0.75)
    sharded, counts = dedup.deduplicate_with_stats(
        docs, exact=True, near=True, threshold=0.75, near_sharded=True
    )
    assert len(in_mem) == len(sharded) == 2
    assert counts["removed_near"] == 1


def test_sharded_validation():
    try:
        dedup.near_duplicate_groups_sharded(
            [{"text": "x"}], num_hashes=10, num_bands=3, rows_per_band=4
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for band mismatch")


def test_dedup_counts_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 5, encoding="utf-8")
    (raw / "b.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 5, encoding="utf-8")
    (raw / "c.txt").write_text("সম্পূর্ণ ভিন্ন কিছু লেখা। " * 5, encoding="utf-8")
    cfg = PipelineConfig(
        raw_dir=str(raw),
        out_root=str(tmp_path / "out"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary = run_pipeline(cfg)
    assert summary["dedup"]["removed_exact"] == 1
    assert summary["after_dedup"] == 2
    assert summary["stats"]["dedup"]["rate"] == 1 / 3


def test_cross_source_dedup_via_pipeline(tmp_path):
    state = tmp_path / "state.txt"
    raw_a = tmp_path / "raw_a"
    raw_a.mkdir()
    (raw_a / "a.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 5, encoding="utf-8")
    cfg_a = PipelineConfig(
        raw_dir=str(raw_a),
        out_root=str(tmp_path / "out_a"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        dedup_state_path=str(state),
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary_a = run_pipeline(cfg_a)
    assert summary_a["after_dedup"] == 1

    raw_b = tmp_path / "raw_b"
    raw_b.mkdir()
    (raw_b / "a.txt").write_text("বাংলা ভাষা একটি সমৃদ্ধ ভাষা। " * 5, encoding="utf-8")
    (raw_b / "new.txt").write_text("একদম নতুন লেখা যা আগে দেখিনি। " * 5, encoding="utf-8")
    cfg_b = PipelineConfig(
        raw_dir=str(raw_b),
        out_root=str(tmp_path / "out_b"),
        min_chars=1,
        max_chars=10**6,
        min_words=1,
        require_bangla=False,
        dedup_state_path=str(state),
        strict=False,  # dev bypass: no source registry configured in this test
    )
    summary_b = run_pipeline(cfg_b)
    assert summary_b["after_dedup"] == 1  # duplicate from source A dropped
    assert summary_b["dedup"]["removed_exact"] == 1

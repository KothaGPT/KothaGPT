"""Tests for the Phase 1B Bangla tokenizer (BPE, Unigram, and WordPiece)."""

from __future__ import annotations

import random

import pytest

from ml.tokenizer import (
    load_tokenizer,
    train_bpe,
    train_unigram,
    train_wordpiece,
)
from ml.tokenizer.benchmark import (
    GATED_SETS,
    SAMPLE_TEXTS,
    check_benchmark,
    run_benchmark,
)
from ml.tokenizer.corpus import load_corpus

_BANGLA_WORDS = [
    "বাংলা",
    "ভাষা",
    "আন্দোলন",
    "সাহিত্য",
    "কবিতা",
    "গান",
    "নদী",
    "পাহাড়",
    "সবুজ",
    "শহর",
    "গ্রাম",
    "মানুষ",
    "ইতিহাস",
    "সংস্কৃতি",
    "শিক্ষা",
    "বিজ্ঞান",
    "প্রযুক্তি",
    "কম্পিউটার",
    "নেটওয়ার্ক",
    "তথ্য",
    "ডেটা",
    "মডেল",
    "অনুবাদ",
    "সামগ্রী",
    "প্রকাশনা",
    "লেখক",
    "পাঠক",
    "বাজার",
    "অর্থনীতি",
    "স্বাধীনতা",
    "সোনালী",
    "আকাশ",
    "মেঘ",
    "বৃষ্টি",
    "জল",
    "সমুদ্র",
    "তরঙ্গ",
    "পাখি",
    "ফুল",
    "গাছ",
    "মাটি",
    "রোদ",
    "চাঁদ",
    "তারকা",
    "রাত",
    "দিন",
    "সকাল",
    "সন্ধ্যা",
    "শিশু",
    "মা",
    "বাবা",
    "ভাই",
    "বোন",
    "পরিবার",
    "বন্ধু",
    "প্রতিবেশী",
    "সহকর্মী",
    "স্কুল",
    "কলেজ",
    "বিশ্ববিদ্যালয়",
    "শিক্ষক",
    "শিক্ষার্থী",
    "পরীক্ষা",
    "ফলাফল",
    "কথা",
    "মতামত",
    "প্রশ্ন",
    "উত্তর",
    "সমস্যা",
    "সমাধান",
    "উন্নয়ন",
    "অগ্রগতি",
    "অর্থ",
    "টাকা",
    "ব্যাংক",
    "লেনদেন",
    "রপ্তানি",
    "আমদানি",
    "শিল্প",
    "কারখানা",
    "কৃষি",
    "ধান",
    "গম",
    "শাকসবজি",
    "ফলমূল",
    "মাছ",
    "মাংস",
    "দুধ",
    "চা",
    "কফি",
    "ভাত",
    "রুটি",
    "তরকারি",
    "মিষ্টি",
    "পিঠা",
    "উৎসব",
    "ঈদ",
    "পূজা",
    "বিয়ে",
    "বই",
    "খবর",
    "সংবাদপত্র",
    "চলচ্চিত্র",
    "নাটক",
    "টেলিভিশন",
    "রেডিও",
    "ফটো",
    "স্মার্টফোন",
    "ইন্টারনেট",
    "ইমেইল",
    "ওয়েবসাইট",
    "অ্যাপ",
    "গেম",
    "ভিডিও",
]


def _make_corpus(n: int = 800, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    corpus = []
    for _ in range(n):
        k = rng.randint(12, 35)
        corpus.append(" ".join(rng.choice(_BANGLA_WORDS) for _ in range(k)))
    return corpus


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    return _make_corpus()


def test_bpe_trains_vocab_and_merges(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    assert len(tokenizer.vocab) >= 300
    assert tokenizer.merges
    # the first merge must have produced a real concatenated token
    first = tokenizer.merges[0]
    assert "".join(first) in tokenizer.vocab


def test_bpe_roundtrip(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    for text in corpus[:20]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unigram_trains_and_roundtrips(corpus):
    tokenizer = train_unigram(corpus, vocab_size=400, min_frequency=2, iterations=3)
    assert tokenizer.probs
    for text in corpus[:20]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_wordpiece_trains_vocab(corpus):
    tokenizer = train_wordpiece(corpus, vocab_size=400, min_frequency=2)
    assert len(tokenizer.vocab) >= 300
    assert tokenizer.merges


def test_wordpiece_roundtrip(corpus):
    tokenizer = train_wordpiece(corpus, vocab_size=400, min_frequency=2)
    for text in corpus[:20]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_contains_special_tokens(corpus):
    tokenizer = train_bpe(corpus, vocab_size=200, min_frequency=2)
    for special in ("<unk>", "<bos>", "<eos>", "<pad>"):
        assert special in tokenizer.vocab


def test_save_load_roundtrip(corpus, tmp_path):
    bpe = train_bpe(corpus, vocab_size=300, min_frequency=2)
    unigram = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    wordpiece = train_wordpiece(corpus, vocab_size=300, min_frequency=2)
    for tokenizer, name in ((bpe, "bpe"), (unigram, "unigram"), (wordpiece, "wordpiece")):
        tokenizer.save(tmp_path / name)
        loaded = load_tokenizer(tmp_path / name / "tokenizer.json")
        assert type(loaded) is type(tokenizer)
        for text in corpus[:10]:
            assert loaded.encode(text) == tokenizer.encode(text)


def test_emoji_punctuation_mixed_code_no_crash(corpus):
    tokenizer = train_bpe(corpus, vocab_size=300, min_frequency=2)
    for text in SAMPLE_TEXTS.values():
        ids = tokenizer.encode(text)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert tokenizer.decode(ids)


def test_unk_reports_coverage(corpus):
    tokenizer = train_bpe(corpus, vocab_size=300, min_frequency=2)
    stats = tokenizer.stats(SAMPLE_TEXTS["bangla"])
    assert stats["tokens"] > 0
    assert 0 <= stats["tokens_per_char"] < 5
    assert "unk" in stats


def test_unseen_bangla_characters_encode(corpus):
    """Characters outside the training corpus must still tokenize (bootstrap)."""
    tokenizer = train_bpe(corpus, vocab_size=200, min_frequency=2)
    text = "এটি এমন একটি বাক্য যাতে এ ও ঊ এবং ঔ আছে"  # letters not in synthetic corpus
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    unigram = train_unigram(corpus, vocab_size=200, min_frequency=2, iterations=2)
    ids = unigram.encode(text)
    assert unigram.decode(ids) == text
    wordpiece = train_wordpiece(corpus, vocab_size=200, min_frequency=2)
    ids = wordpiece.encode(text)
    assert wordpiece.decode(ids) == text


def test_benchmark_shape(corpus):
    tokenizer = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    result = run_benchmark(tokenizer)
    assert result["avg_tokens_per_char"] > 0
    assert set(result["per_set"]) == {
        "bangla",
        "mixed",
        "punctuation",
        "emoji",
        "code",
        "translit",
        "digits",
        "names",
        "social",
    }
    assert result["dev_max_unk_rate"] >= 0.0
    assert result["dev_min_decode_fidelity"] >= 0.0


def test_benchmark_gate_passes_on_bangla_dev(corpus):
    tokenizer = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    result = run_benchmark(tokenizer)
    failures = check_benchmark(result)
    assert failures == [], failures


def test_benchmark_gate_detects_unk_regression(corpus):
    tokenizer = train_unigram(corpus, vocab_size=300, min_frequency=2, iterations=2)
    result = run_benchmark(tokenizer)
    # An impossible unk threshold must be reported as a violation.
    failures = check_benchmark(result, {"max_dev_unk_rate": -1.0})
    assert any("dev_max_unk_rate" in f for f in failures)


def test_benchmark_decodes_dev_sets_losslessly(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    for name in ("bangla", "mixed", "punctuation", "digits", "names"):
        result = run_benchmark(tokenizer, {name: SAMPLE_TEXTS[name]})
        assert result["per_set"][name]["decode_fidelity"] == 1.0


def test_translit_set_is_preprocessed(corpus):
    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    stats = run_benchmark(tokenizer)["per_set"]["translit"]
    # The romanized sample must be transliterated to Bangla before tokenizing,
    # so it must contain zero unknowns and round-trip cleanly.
    assert stats["unk_rate"] == 0.0
    assert stats["decode_fidelity"] == 1.0


def test_gated_sets_exclude_emoji_stress(corpus):
    assert "emoji" not in GATED_SETS
    assert "social" not in GATED_SETS
    assert "bangla" in GATED_SETS


def test_unigram_improves_over_initial_vocab(corpus):
    tokenizer = train_unigram(corpus, vocab_size=400, min_frequency=2, iterations=3)
    assert len(tokenizer.vocab) >= 300


def test_load_corpus_jsonl(tmp_path):
    import json

    (tmp_path / "c.jsonl").write_text(
        json.dumps({"text": "বাংলা ভাষা"}) + "\n" + json.dumps({"text": "সাহিত্য ও সংস্কৃতি"}) + "\n",
        encoding="utf-8",
    )
    texts = load_corpus(tmp_path / "c.jsonl")
    assert texts == ["বাংলা ভাষা", "সাহিত্য ও সংস্কৃতি"]


# --- WS-1 / WS-6 vocabulary freeze ----------------------------------------


def test_corpus_digest_is_content_addressed():
    from ml.tokenizer import corpus_digest

    a = corpus_digest(["বাংলা ভাষা", "সাহিত্য"])
    b = corpus_digest(["বাংলা ভাষা"])
    c = corpus_digest(["বাংলা ভাষা", "সাহিত্য"])
    assert a == c
    assert a != b
    assert len(a) == 12


def test_version_id_includes_corpus_and_vocab():
    from ml.tokenizer import version_id

    v1 = version_id("cafe123", {"<unk>": 0, "ব": 1})
    v2 = version_id("cafe124", {"<unk>": 0, "ব": 1})
    v3 = version_id("cafe123", {"<unk>": 0, "ব": 2})
    assert v1.startswith("1.0.0+cafe123.")
    assert v1 != v2
    assert v1 != v3


def test_coverage_report_metrics(corpus):
    from ml.tokenizer import coverage_report

    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    report = coverage_report(tokenizer, corpus)
    assert 0.0 <= report["coverage"] <= 1.0
    assert 0.0 <= report["unk_rate"] <= 1.0
    assert report["tokens_per_char"] > 0.0
    assert report["num_known"] > 0


def test_coverage_report_full_coverage_on_seen_text(corpus):
    from ml.tokenizer import coverage_report

    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=1)
    # Every char of the training corpus is in BOOTSTRAP_CHARS, so the
    # tokenizer must cover the same text losslessly.
    report = coverage_report(tokenizer, corpus[:10])
    assert report["unk_rate"] == 0.0
    assert report["coverage"] == 1.0


def test_coverage_report_whitespace_not_penalized(corpus):
    from ml.tokenizer import coverage_report

    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=1)
    # Newlines / tabs are encoded as word-marker tokens (not script chars), so
    # they must not depress coverage. Regression for the 98.5%-vs-100% gap that
    # was caused by counting U+000A in the denominator.
    lines = [corpus[0], corpus[1], corpus[2]]
    single = coverage_report(tokenizer, lines)
    multi = coverage_report(tokenizer, ["\n".join(lines)])
    assert single["coverage"] == multi["coverage"] == 1.0


def test_export_vocab_matches_tokenizer(corpus):
    from ml.tokenizer import export_vocab

    tokenizer = train_bpe(corpus, vocab_size=400, min_frequency=2)
    exported = export_vocab(tokenizer)
    assert exported == tokenizer.vocab
    assert "<unk>" in exported

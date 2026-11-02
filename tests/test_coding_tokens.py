from __future__ import annotations

import pytest

from ml.tokenizer.bpe import BpeTokenizer, train_bpe
from ml.tokenizer.coding import (
    CODING_SPECIAL_TOKENS,
    CodingTokenExtensions,
    apply_coding_extensions,
    coding_extensions_digest,
)


def _train_tokenizer(vocab_size: int = 64) -> BpeTokenizer:
    corpus = [
        "def add(a, b): return a + b",
        "def square(n): return n * n",
        "here is a bug in the code",
        "বাংলা মন্তব্য: এটি একটি ফাংশন",
        "explain the function please",
    ]
    return train_bpe(corpus, vocab_size=vocab_size, min_frequency=1)


def test_coding_special_tokens_are_unique_and_ordered():
    assert len(CODING_SPECIAL_TOKENS) == len(set(CODING_SPECIAL_TOKENS))
    assert CODING_SPECIAL_TOKENS[:2] == ["<code>", "</code>"]
    assert "<function>" in CODING_SPECIAL_TOKENS
    assert "<bangla_comment>" in CODING_SPECIAL_TOKENS


def test_coding_extensions_default_offset_is_base_vocab_size():
    tok = _train_tokenizer()
    ext = CodingTokenExtensions(base_vocab_size=len(tok.vocab))
    ids = ext.token_ids()
    assert ids["<code>"] == len(tok.vocab)
    assert ids["</code>"] == len(tok.vocab) + 1
    assert len(ids) == len(CODING_SPECIAL_TOKENS)


def test_coding_extensions_reject_empty_or_duplicate_tokens():
    with pytest.raises(ValueError, match="tokens must be non-empty"):
        CodingTokenExtensions(tokens=[])
    with pytest.raises(ValueError, match="coding tokens must be unique"):
        CodingTokenExtensions(tokens=["<code>", "<code>"])


def test_coding_extensions_reject_negative_base_vocab_size():
    with pytest.raises(ValueError, match="base_vocab_size must be non-negative"):
        CodingTokenExtensions(base_vocab_size=-1)


def test_coding_extensions_digest_is_stable_and_hex():
    ext = CodingTokenExtensions(base_vocab_size=128)
    a = ext.digest()
    b = CodingTokenExtensions(base_vocab_size=128).digest()
    assert a == b
    assert len(a) == 64
    assert coding_extensions_digest(ext) == a


def test_coding_extensions_to_from_dict_round_trip():
    ext = CodingTokenExtensions(base_vocab_size=256, version="v2")
    restored = CodingTokenExtensions.from_dict(ext.to_dict())
    assert restored.base_vocab_size == 256
    assert restored.version == "v2"
    assert restored.tokens == ext.tokens


def test_apply_coding_extensions_returns_token_ids():
    tok = _train_tokenizer()
    ids = apply_coding_extensions(tok)
    assert ids["<code>"] == len(tok.vocab)
    assert ids["<bangla_comment>"] == len(tok.vocab) + 8


def test_apply_coding_extensions_rejects_tokenizer_without_vocab():
    class Fake:
        pass

    with pytest.raises(TypeError, match=".vocab dict"):
        apply_coding_extensions(Fake())


def test_coding_extensions_do_not_mutate_base_tokenizer():
    tok = _train_tokenizer()
    base_size = len(tok.vocab)
    apply_coding_extensions(tok)
    assert len(tok.vocab) == base_size
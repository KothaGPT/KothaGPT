"""Unigram (sentencepiece-style) tokenizer: trainer and encoder.

Training starts from all corpus substrings above a frequency floor, then runs
EM: the Viterbi best segmentation of each word contributes expected counts,
probabilities are re-estimated, and low-probability tokens are pruned until the
target vocabulary size is reached. Encoding is a Viterbi shortest path over a
trie of the final vocabulary.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from .base import _WORD_MARKER, BOOTSTRAP_CHARS, SPECIAL_TOKENS, UNK, BaseTokenizer

__all__ = ["UnigramTokenizer", "train_unigram"]


def _build_trie(tokens: list[str]) -> dict:
    root: dict = {"__tok__": None}
    for token in tokens:
        node = root
        for ch in token:
            node = node.setdefault(ch, {"__tok__": None})
        node["__tok__"] = token
    return root


def _best_segmentation(word: str, trie: dict, neg_log: dict[str, float]) -> list[str]:
    """Viterbi: split ``word`` into the minimum -log-probability token path."""
    n = len(word)
    scores = [float("inf")] * (n + 1)
    path: list[list[str]] = [[] for _ in range(n + 1)]
    scores[0] = 0.0
    for i in range(n):
        if scores[i] == float("inf"):
            continue
        node = trie
        for j in range(i, n):
            ch = word[j]
            if ch not in node:
                break
            node = node[ch]
            token = node["__tok__"]
            if token is not None:
                score = scores[i] + neg_log[token]
                if score < scores[j + 1]:
                    scores[j + 1] = score
                    path[j + 1] = path[i] + [token]
    if scores[n] == float("inf"):
        return [UNK]
    return path[n]


def train_unigram(
    texts: list[str],
    vocab_size: int,
    *,
    min_frequency: int = 2,
    max_subword_len: int = 8,
    iterations: int = 8,
    log: Callable[[str], None] | None = None,
) -> UnigramTokenizer:
    """Train a Unigram tokenizer from a list of texts."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must exceed {len(SPECIAL_TOKENS)} special tokens")
    word_counts: Counter[str] = Counter()
    for text in texts:
        for word in text.split():
            word_counts[_WORD_MARKER + word] += 1
    if not word_counts:
        raise ValueError("empty corpus: no words to train on")

    substring_counts: Counter[str] = Counter()
    for word, cnt in word_counts.items():
        n = len(word)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_subword_len) + 1):
                substring_counts[word[i:j]] += cnt

    chars: set[str] = set()
    for word in word_counts:
        chars.update(word)
    chars |= set(BOOTSTRAP_CHARS)

    vocab: set[str] = {token for token, cnt in substring_counts.items() if cnt >= min_frequency}
    vocab |= chars
    if len(vocab) < len(SPECIAL_TOKENS):
        raise ValueError("corpus vocabulary smaller than special-token count")

    total = sum(substring_counts[token] for token in vocab if token in substring_counts)
    probs: dict[str, float] = {token: substring_counts.get(token, 1) / total for token in vocab}
    target = vocab_size - len(SPECIAL_TOKENS)
    _EPS = 1e-9

    for epoch in range(iterations):
        neg_log = {t: -math.log(max(p, _EPS)) for t, p in probs.items()}
        trie = _build_trie(list(probs))
        usage: Counter[str] = Counter()
        for word, cnt in word_counts.items():
            for token in _best_segmentation(word, trie, neg_log):
                usage[token] += cnt
        # Re-estimate probabilities over the CURRENT vocabulary. Tokens that
        # never win a Viterbi path keep a tiny probability instead of vanishing,
        # so the vocabulary stays stable between epochs.
        smoothed = {token: usage.get(token, _EPS) for token in probs}
        used_total = sum(smoothed.values())
        probs = {token: count / used_total for token, count in smoothed.items()}
        if len(probs) > target:
            removable = [t for t in probs if t not in chars]
            removable.sort(key=lambda t: probs[t])
            for token in removable[: len(probs) - target]:
                del probs[token]
        if log is not None:
            log(f"unigram: epoch {epoch + 1}/{iterations}, vocab={len(probs)}")

    keep = set(probs) | chars
    probs = {t: probs.get(t, _EPS) for t in keep}
    weight = sum(probs.values())
    probs = {t: p / weight for t, p in probs.items()}

    vocab_ids: dict[str, int] = {}
    vocab_ids.update(SPECIAL_TOKENS)
    for offset, token in enumerate(sorted(keep)):
        vocab_ids[token] = len(SPECIAL_TOKENS) + offset
    return UnigramTokenizer(vocab_ids, probs)


class UnigramTokenizer(BaseTokenizer):
    """Unigram tokenizer; encodes via Viterbi over the vocab trie."""

    type = "unigram"

    def __init__(self, vocab: dict[str, int], probs: dict[str, float]) -> None:
        super().__init__(vocab)
        self.probs = {t: probs.get(t, 1e-12) for t in vocab}
        self.neg_log = {t: -math.log(max(p, 1e-12)) for t, p in self.probs.items()}
        self._trie = _build_trie(list(self.probs))

    def _encode_word(self, word: str) -> list[str]:
        return _best_segmentation(word, self._trie, self.neg_log)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["probs"] = self.probs
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnigramTokenizer:
        vocab = data["vocab"]
        probs = data.get("probs", {})
        if UNK not in vocab:
            raise ValueError("saved Unigram tokenizer missing <unk> special token")
        return cls(vocab, probs)

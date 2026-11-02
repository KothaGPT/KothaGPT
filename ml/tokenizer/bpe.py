"""Byte-pair encoding (BPE) tokenizer: trainer and encoder.

Training uses the classic word-frequency-counted algorithm (subword-nmt
style): each distinct word keeps its current symbol segmentation, pair counts
are aggregated over words weighted by frequency, and the most frequent pair is
merged each step. A lazy heap keeps the argmax cheap.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from .base import _WORD_MARKER, BOOTSTRAP_CHARS, SPECIAL_TOKENS, UNK, BaseTokenizer

__all__ = ["BpeTokenizer", "train_bpe"]


def _merge_all(symbols: tuple[str, ...], a: str, b: str, merged: str) -> tuple[str, ...]:
    out: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(
    texts: list[str],
    vocab_size: int,
    *,
    min_frequency: int = 2,
    log: Callable[[str], None] | None = None,
) -> BpeTokenizer:
    """Train a BPE tokenizer from a list of texts."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must exceed {len(SPECIAL_TOKENS)} special tokens")
    word_counts: Counter[str] = Counter()
    for text in texts:
        for word in text.split():
            word_counts[_WORD_MARKER + word] += 1
    if not word_counts:
        raise ValueError("empty corpus: no words to train on")

    symbols: dict[str, tuple[str, ...]] = {w: tuple(w) for w in word_counts}
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_to_words: dict[tuple[str, str], set[str]] = defaultdict(set)
    for word, sym in symbols.items():
        cnt = word_counts[word]
        for pair in pairwise(sym):
            pair_counts[pair] += cnt
            pair_to_words[pair].add(word)

    vocab: set[str] = set()
    for word in word_counts:
        vocab.update(word)
    vocab |= set(BOOTSTRAP_CHARS)
    if len(vocab) < len(SPECIAL_TOKENS):
        raise ValueError("corpus vocabulary smaller than special-token count")

    heap: list[tuple[int, tuple[str, str]]] = []
    for pair, cnt in pair_counts.items():
        heapq.heappush(heap, (-cnt, pair))

    merges: list[tuple[str, str]] = []
    target = vocab_size - len(SPECIAL_TOKENS)
    while len(vocab) < target:
        best = None
        while heap:
            neg_cnt, pair = heap[0]
            if pair_counts.get(pair, 0) != -neg_cnt:
                heapq.heappop(heap)
                continue
            best = pair
            break
        if best is None:
            break
        a, b = best
        cnt = pair_counts[best]
        if cnt < min_frequency:
            break
        merged = a + b
        vocab.add(merged)
        merges.append(best)

        for word in list(pair_to_words.get(best, ())):
            sym = symbols[word]
            wcnt = word_counts[word]
            old_pairs = Counter(pairwise(sym))
            new_sym = _merge_all(sym, a, b, merged)
            symbols[word] = new_sym
            for pair, multiplicity in old_pairs.items():
                pair_counts[pair] -= wcnt * multiplicity
                pair_to_words[pair].discard(word)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)
                else:
                    heapq.heappush(heap, (-pair_counts[pair], pair))
            new_pairs = Counter(pairwise(new_sym))
            for pair, multiplicity in new_pairs.items():
                pair_counts[pair] += wcnt * multiplicity
                pair_to_words[pair].add(word)
                heapq.heappush(heap, (-pair_counts[pair], pair))
        if len(heap) > 8 * max(len(pair_counts), 1):
            heap = [(-cnt, pair) for pair, cnt in pair_counts.items()]
            heapq.heapify(heap)
        if log is not None and len(merges) % 1000 == 0:
            log(f"bpe: {len(merges)} merges, vocab={len(vocab)}")

    vocab_ids: dict[str, int] = {}
    vocab_ids.update(SPECIAL_TOKENS)
    for offset, token in enumerate(sorted(vocab)):
        vocab_ids[token] = len(SPECIAL_TOKENS) + offset
    return BpeTokenizer(vocab_ids, merges)


class BpeTokenizer(BaseTokenizer):
    """BPE tokenizer; decodes by repeatedly merging the lowest-rank pair."""

    type = "bpe"

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        super().__init__(vocab)
        self.merges = merges
        self.bpe_ranks: dict[tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

    def _encode_word(self, word: str) -> list[str]:
        tokens = tuple(word)
        if len(tokens) <= 1:
            return [word]
        while True:
            best_pair: tuple[str, str] | None = None
            best_rank = len(self.merges)
            for pair in pairwise(tokens):
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            tokens = _merge_all(tokens, a, b, merged)
        return list(tokens)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["merges"] = [list(pair) for pair in self.merges]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BpeTokenizer:
        vocab = data["vocab"]
        merges = [tuple(pair) for pair in data.get("merges", [])]
        if UNK not in vocab:
            raise ValueError("saved BPE tokenizer missing <unk> special token")
        return cls(vocab, merges)

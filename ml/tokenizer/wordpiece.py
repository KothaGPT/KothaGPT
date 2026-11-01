"""WordPiece tokenizer: trainer and encoder.

Training selects merges that maximise the likelihood increase, i.e. the pair
(a, b) with the highest ``freq(ab) / (freq(a) * freq(b))``.  Encoding uses a
greedy longest-match-first scan over the merged subwords in the vocabulary.

This implementation follows the existing BPE/Unigram conventions: word-based
tokenisation with a leading ``▁`` (U+2581) space marker and the full Bengali
block bootstrapped into every vocabulary.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from .base import _WORD_MARKER, BOOTSTRAP_CHARS, SPECIAL_TOKENS, UNK, BaseTokenizer

__all__ = ["WordPieceTokenizer", "train_wordpiece"]


def train_wordpiece(
    texts: list[str],
    vocab_size: int,
    *,
    min_frequency: int = 2,
    log: Callable[[str], None] | None = None,
) -> WordPieceTokenizer:
    """Train a WordPiece tokenizer from a list of texts."""
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must exceed {len(SPECIAL_TOKENS)} special tokens")
    word_counts: Counter[str] = Counter()
    for text in texts:
        for word in text.split():
            word_counts[_WORD_MARKER + word] += 1
    if not word_counts:
        raise ValueError("empty corpus: no words to train on")

    # Start with individual characters as base vocabulary.
    vocab: set[str] = set()
    for word in word_counts:
        vocab.update(word)
    vocab |= set(BOOTSTRAP_CHARS)

    # Track how each word is segmented and compute symbol-pair frequencies.
    symbols: dict[str, tuple[str, ...]] = {w: tuple(w) for w in word_counts}
    word_order: dict[str, int] = {w: i for i, w in enumerate(word_counts)}
    pair_counts: Counter[tuple[str, str]] = Counter()
    symbol_counts: Counter[str] = Counter()
    pair_words: dict[tuple[str, str], set[str]] = {}
    symbol_pairs: dict[str, set[tuple[str, str]]] = {}
    pair_order: dict[tuple[str, str], int] = {}
    next_pair_order = 0

    def add_pair(pair: tuple[str, str], word: str, count: int) -> None:
        nonlocal next_pair_order
        if pair not in pair_counts:
            pair_counts[pair] = 0
            pair_order[pair] = next_pair_order
            next_pair_order += 1
            pair_words[pair] = set()
            for symbol in pair:
                symbol_pairs.setdefault(symbol, set()).add(pair)
        pair_counts[pair] += count
        pair_words[pair].add(word)

    for word, sym in symbols.items():
        cnt = word_counts[word]
        for s in sym:
            symbol_counts[s] += cnt
        for pair in pairwise(sym):
            add_pair(pair, word, cnt)

    def pair_score(pair: tuple[str, str]) -> float | None:
        freq = pair_counts.get(pair, 0)
        if freq < min_frequency:
            return None
        a, b = pair
        denom = symbol_counts.get(a, 0) * symbol_counts.get(b, 0)
        if denom == 0:
            return None
        return freq / denom

    scores: dict[tuple[str, str], float] = {}
    heap: list[tuple[float, int, tuple[str, str]]] = []
    for pair in pair_counts:
        score = pair_score(pair)
        if score is not None:
            scores[pair] = score
            heapq.heappush(heap, (-score, pair_order[pair], pair))

    target = vocab_size - len(SPECIAL_TOKENS)
    merges: list[tuple[str, str]] = []

    while len(vocab) < target:
        # Pick the pair maximising freq(ab) / (freq(a) * freq(b)).
        best_pair: tuple[str, str] | None = None
        while heap:
            neg_score, order, pair = heapq.heappop(heap)
            if pair_order.get(pair) != order:
                continue
            current = scores.get(pair)
            if current is None or current != -neg_score:
                continue
            best_pair = pair
            break
        if best_pair is None:
            break

        a, b = best_pair
        merged = a + b
        vocab.add(merged)
        merges.append(best_pair)

        # Apply the merge to every word that contained this pair.
        affected = sorted(pair_words.get(best_pair, ()), key=word_order.__getitem__)
        impacted: set[tuple[str, str]] = set()
        changed_symbols: set[str] = set()
        for word in affected:
            sym = symbols[word]
            new_sym: list[str] = []
            i = 0
            wcnt = word_counts[word]
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    new_sym.append(merged)
                    i += 2
                else:
                    new_sym.append(sym[i])
                    i += 1
            new_sym_t = tuple(new_sym)
            if new_sym_t == sym:
                continue
            old_pairs = list(pairwise(sym))
            new_pairs = list(pairwise(new_sym_t))
            impacted.update(old_pairs)
            impacted.update(new_pairs)
            changed_symbols.update(sym)
            changed_symbols.update(new_sym_t)
            # Update counts.
            for s in sym:
                symbol_counts[s] -= wcnt
            for pair in old_pairs:
                pair_counts[pair] -= wcnt
                pair_words[pair].discard(word)
            for s in new_sym_t:
                symbol_counts[s] += wcnt
            for pair in new_pairs:
                add_pair(pair, word, wcnt)
            symbols[word] = new_sym_t

        for symbol in changed_symbols:
            impacted.update(symbol_pairs.get(symbol, ()))

        # Clean up stale entries.
        for pair in impacted:
            if pair_counts.get(pair, 0) <= 0:
                del pair_counts[pair]
                pair_order.pop(pair, None)
                pair_words.pop(pair, None)
                scores.pop(pair, None)
                for symbol in pair:
                    pairs = symbol_pairs.get(symbol)
                    if pairs is not None:
                        pairs.discard(pair)
                        if not pairs:
                            del symbol_pairs[symbol]

        for pair in impacted:
            if pair not in pair_counts:
                continue
            score = pair_score(pair)
            if score is None:
                scores.pop(pair, None)
            else:
                scores[pair] = score
                heapq.heappush(heap, (-score, pair_order[pair], pair))

        if log is not None and len(merges) % 1000 == 0:
            log(f"wordpiece: {len(merges)} merges, vocab={len(vocab)}")

    vocab_ids: dict[str, int] = {}
    vocab_ids.update(SPECIAL_TOKENS)
    for offset, token in enumerate(sorted(vocab)):
        vocab_ids[token] = len(SPECIAL_TOKENS) + offset
    return WordPieceTokenizer(vocab_ids, merges)


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer; encodes via greedy longest-match-first."""

    type = "wordpiece"

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        super().__init__(vocab)
        self.merges = merges
        self._max_token_len = max(len(t) for t in vocab)

    def _encode_word(self, word: str) -> list[str]:
        tokens: list[str] = []
        start = 0
        n = len(word)
        while start < n:
            end = min(start + self._max_token_len, n)
            found = False
            while start < end:
                substr = word[start:end]
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append(UNK)
                start += 1
            else:
                start = end
        return tokens

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["merges"] = [list(pair) for pair in self.merges]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordPieceTokenizer:
        vocab = data["vocab"]
        merges = [tuple(pair) for pair in data.get("merges", [])]
        if UNK not in vocab:
            raise ValueError("saved WordPiece tokenizer missing <unk> special token")
        return cls(vocab, merges)

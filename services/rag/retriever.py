from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .chunk import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()


def _terms(value: str) -> list[str]:
    return re.findall(r"[\w\u0980-\u09ff]+", value.casefold())


class LexicalRetriever:
    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._chunks: list[Chunk] = list(chunks)
        self._df = Counter(term for chunk in self._chunks for term in set(_terms(chunk.content)))

    def add(self, chunks: Iterable[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._df = Counter(term for chunk in self._chunks for term in set(_terms(chunk.content)))

    def search(
        self, query: str, *, top_k: int = 5, document_id: str | None = None
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = set(_terms(query))
        if not query_terms:
            return []
        total = max(1, len(self._chunks))
        results: list[SearchResult] = []
        for chunk in self._chunks:
            if document_id and chunk.document_id != document_id:
                continue
            counts = Counter(_terms(chunk.content))
            matched = tuple(sorted(query_terms & counts.keys()))
            if not matched:
                continue
            score = sum(
                (1 + math.log1p(counts[t])) * math.log((total + 1) / (self._df[t] + 1))
                for t in matched
            )
            results.append(SearchResult(chunk, score, matched))
        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk.document_id, result.chunk.index),
        )[:top_k]

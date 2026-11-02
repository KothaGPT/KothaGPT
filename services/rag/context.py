from __future__ import annotations

from dataclasses import dataclass

from .retriever import LexicalRetriever, SearchResult


@dataclass(frozen=True)
class Citation:
    citation_id: str
    source: str
    document_id: str
    chunk_id: str
    start: int
    end: int


@dataclass(frozen=True)
class GroundedContext:
    query: str
    text: str
    results: tuple[SearchResult, ...]
    citations: tuple[Citation, ...]


def build_context(
    query: str, retriever: LexicalRetriever, *, top_k: int = 5, max_chars: int = 4000
) -> GroundedContext:
    selected: list[SearchResult] = []
    used = 0
    for result in retriever.search(query, top_k=top_k):
        addition = len(result.chunk.content) + 16
        if selected and used + addition > max_chars:
            break
        selected.append(result)
        used += addition
    citations = tuple(
        Citation(
            f"[{i + 1}]",
            r.chunk.source,
            r.chunk.document_id,
            r.chunk.chunk_id,
            r.chunk.start,
            r.chunk.end,
        )
        for i, r in enumerate(selected)
    )
    text = "\n\n".join(
        f"{citation.citation_id} {result.chunk.content}"
        for citation, result in zip(citations, selected)
    )
    return GroundedContext(query, text, tuple(selected), citations)

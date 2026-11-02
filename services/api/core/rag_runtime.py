"""Shared RAG runtime singleton used by the API routers.

Holds one ``VectorStore``, one ``LexicalRetriever`` and one ``IngestPipeline``
so that documents indexed via ``/v1/rag/ingest`` are searchable both through
the dedicated search endpoints and through grounded chat generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from services.rag.chunk import chunk_text
from services.rag.context import build_context
from services.rag.ingest import DocumentRecord, IngestPipeline
from services.rag.retriever import LexicalRetriever
from services.rag.store import VectorStore

_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
_RAG_MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "4000"))


@dataclass
class RAGRuntime:
    store: VectorStore = field(default_factory=lambda: VectorStore(collection="kothagpt"))
    retriever: LexicalRetriever = field(default_factory=LexicalRetriever)
    pipeline: IngestPipeline = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = IngestPipeline(retriever=self.retriever)

    def ingest(
        self,
        text: str,
        *,
        document_id: str | None = None,
        source: str = "api",
        license: str | None = None,
    ) -> DocumentRecord:
        record = self.pipeline.ingest_text(
            text, source=source, document_id=document_id, license=license
        )
        chunks = chunk_text(text, document_id=record.document_id, source=source)
        if chunks:
            self.store.upsert(chunks)
            self.retriever.add(chunks)
        return record

    def search(self, query: str, top_k: int = _RAG_TOP_K) -> list[dict[str, Any]]:
        results = self.retriever.search(query, top_k=top_k)
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.document_id,
                "content": r.chunk.content,
                "score": round(r.score, 4),
                "source": r.chunk.source,
                "index": r.chunk.index,
            }
            for r in results
        ]

    def build_context(
        self, query: str, *, top_k: int = _RAG_TOP_K, max_chars: int = _RAG_MAX_CHARS
    ) -> tuple[str, list[dict[str, Any]]]:
        grounded = build_context(query, self.retriever, top_k=top_k, max_chars=max_chars)
        citations = [
            {
                "citation_id": c.citation_id,
                "source": c.source,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "start": c.start,
                "end": c.end,
            }
            for c in grounded.citations
        ]
        return grounded.text, citations

    def kb_stats(self) -> dict[str, Any]:
        docs = len(self.pipeline._docs)
        chunks = sum(len(rec.chunk_ids) for rec in self.pipeline._docs.values())
        return {
            "id": "default",
            "name": "default",
            "status": "ready" if docs else "empty",
            "documents": docs,
            "chunks": chunks,
            "source": "api",
        }


rag_runtime = RAGRuntime()

__all__ = ["RAGRuntime", "rag_runtime"]

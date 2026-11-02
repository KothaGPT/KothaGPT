"""WS-6 / WS-9 RAG integration for the Kotha GPT API.

Wires the standalone retrieval-augmented generation modules in
``services/rag`` (chunking, lexical/vector store, ingestion, grounded context)
into the HTTP runtime so documents can be indexed and retrieved at request
time. A single default knowledge base is exposed; multi-tenant KB management
can be layered on top once ``packages/core`` defines a KB record type.

Endpoints:

- ``POST /v1/rag/ingest`` — index a piece of text into the default KB.
  Idempotent by content digest.
- ``POST /v1/rag/search`` — lexical + vector search over the indexed corpus.
- ``POST /v1/rag/context`` — build a grounded, citation-bearing context block
  for a query (suitable for prefixing into a chat prompt).
- ``GET /v1/rag/kb`` — describe the default knowledge base.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...core.rag_runtime import rag_runtime

router = APIRouter(prefix="/v1/rag", tags=["rag"])


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document text to index.")
    document_id: str | None = Field(
        default=None, description="Explicit document id (auto-derived if omitted)."
    )
    source: str = Field(default="api", description="Source label for provenance.")
    license: str | None = Field(default=None, description="Optional license/rights note.")


class IngestResponse(BaseModel):
    document_id: str
    source: str
    digest: str
    status: str
    chunks: int
    chunk_ids: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str
    index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class ContextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=4000, ge=256)


class ContextResponse(BaseModel):
    query: str
    text: str
    citations: list[dict[str, Any]]


class KbResponse(BaseModel):
    id: str
    name: str
    status: str
    documents: int
    chunks: int
    source: str


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Index ``request.text`` into the default knowledge base.

    Re-ingesting the same content digest is a no-op (returns the existing
    record), making this safe to retry.
    """
    record = rag_runtime.ingest(
        request.text,
        document_id=request.document_id,
        source=request.source,
        license=request.license,
    )
    return IngestResponse(
        document_id=record.document_id,
        source=record.source,
        digest=record.digest,
        status=record.status,
        chunks=len(record.chunk_ids),
        chunk_ids=list(record.chunk_ids),
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Search the indexed corpus with the hybrid lexical retriever."""
    results = rag_runtime.search(request.query, top_k=request.top_k)
    items = [
        SearchResultItem(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            content=r["content"],
            score=r["score"],
            source=r["source"],
            index=r["index"],
        )
        for r in results
    ]
    return SearchResponse(query=request.query, results=items)


@router.post("/context", response_model=ContextResponse)
def context(request: ContextRequest) -> ContextResponse:
    """Build a grounded, citation-bearing context block for ``query``."""
    text, citations = rag_runtime.build_context(
        request.query, top_k=request.top_k, max_chars=request.max_chars
    )
    return ContextResponse(query=request.query, text=text, citations=citations)


@router.get("/kb", response_model=KbResponse)
def kb() -> KbResponse:
    """Describe the default knowledge base."""
    stats = rag_runtime.kb_stats()
    return KbResponse(**stats)


__all__ = ["router"]

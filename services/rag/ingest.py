"""WS-1 Document ingestion — dependency-light foundation slice.

Takes files/dirs, normalizes via data.pipeline (reused), chunks via chunk_text,
and indexes via LexicalRetriever. Idempotent by digest, tracks per-document
record in memory (Postgres when DATABASE_URL present).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chunk import chunk_text
from .retriever import LexicalRetriever

try:
    from data.pipeline.normalize import normalize_text as _normalize  # reuse Phase-B
except Exception:  # noqa: BLE001 — fallback if pipeline not importable in minimal env
    def _normalize(text: str) -> str:  # type: ignore
        return text


@dataclass
class DocumentRecord:
    document_id: str
    source: str
    digest: str
    status: str = "ready"
    license: str | None = None
    fetched_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    chunk_ids: list[str] = field(default_factory=list)


class IngestPipeline:
    def __init__(self, retriever: LexicalRetriever | None = None):
        self.retriever = retriever or LexicalRetriever()
        self._docs: dict[str, DocumentRecord] = {}
        self._by_digest: dict[str, str] = {}

    def _digest(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def ingest_text(self, text: str, *, source: str, document_id: str | None = None, license: str | None = None) -> DocumentRecord:
        normalized = _normalize(text)
        digest = self._digest(normalized)
        if digest in self._by_digest:
            # idempotent: already ingested
            return self._docs[self._by_digest[digest]]
        doc_id = document_id or f"doc_{digest}"
        try:
            chunks = chunk_text(normalized, document_id=doc_id, source=source)
            # apply Phase-B gates where sensible (e.g., skip empty)
            if not chunks:
                raise ValueError("no chunks produced")
            self.retriever.add(chunks)
            rec = DocumentRecord(document_id=doc_id, source=source, digest=digest, license=license, chunk_ids=[c.chunk_id for c in chunks])
            self._docs[doc_id] = rec
            self._by_digest[digest] = doc_id
            return rec
        except Exception as e:  # noqa: BLE001 — record failure and continue
            rec = DocumentRecord(document_id=doc_id, source=source, digest=digest, status=f"failed: {e}", license=license)
            self._docs[doc_id] = rec
            return rec

    def ingest_file(self, path: str | Path, *, license: str | None = None) -> DocumentRecord:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Use digest-derived ID (via ingest_text) or full-path hash to avoid collisions on basename
        doc_id = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16]
        return self.ingest_text(text, source=str(p), document_id=f"doc_{doc_id}", license=license)

    def ingest_path(self, path: str | Path, *, pattern: str = "**/*", license: str | None = None) -> list[DocumentRecord]:
        p = Path(path)
        if p.is_file():
            return [self.ingest_file(p, license=license)]
        recs: list[DocumentRecord] = []
        for f in p.glob(pattern):
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".json", ".jsonl"}:
                recs.append(self.ingest_file(f, license=license))
        return recs

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._docs.get(document_id)

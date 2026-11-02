"""WS-6 Vector store — Qdrant + in-memory fallback for CI."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .chunk import Chunk


@dataclass
class StoredChunk:
    chunk: Chunk
    vector: list[float]

class VectorStore:
    def __init__(self, collection: str = "kothagpt", dim: int = 256, url: str | None = None):
        self.collection = collection
        self.dim = dim
        self.url = url or "http://localhost:6333"
        self._mem: list[StoredChunk] = []
        self._client = None
        # Lazy Qdrant client if available
        try:
            from qdrant_client import QdrantClient  # type: ignore
            self._client = QdrantClient(url=self.url)
            # ensure collection exists (tolerant)
            try:
                self._client.get_collection(collection)
            except Exception:  # noqa: BLE001 — collection may not exist yet
                try:
                    self._client.create_collection(collection, vectors_config={"size": dim, "distance": "Cosine"})
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001 — Qdrant unavailable; use in-memory fallback
            self._client = None

    def _embed(self, text: str) -> list[float]:
        # dependency-light hash embedding (same as MockBackend) for CI without model
        import hashlib
        vec = []
        for i in range(self.dim):
            digest = hashlib.sha256(f"{i}:{text}".encode()).digest()
            val = int.from_bytes(digest[:8], "big") / (2**64) * 2 - 1
            vec.append(val)
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [round(v/norm, 6) for v in vec]

    def _stable_id(self, chunk_id: str) -> int:
        import hashlib

        # Deterministic 63-bit ID from sha256, not Python hash()
        return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16) % (2**63)

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        # Deduplicate by chunk_id in memory
        by_id: dict[str, StoredChunk] = {sc.chunk.chunk_id: sc for sc in self._mem}
        vecs = []
        for c in chunks:
            v = self._embed(c.content)
            sc = StoredChunk(c, v)
            by_id[c.chunk_id] = sc
            vecs.append((c.chunk_id, v, {"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source, "content": c.content, "index": c.index, "section": c.section or "", "start": c.start, "end": c.end}))
        self._mem = list(by_id.values())
        if self._client:
            try:
                from qdrant_client.models import PointStruct

                points = [PointStruct(id=self._stable_id(cid), vector=vec, payload=pay) for cid, vec, pay in vecs]
                self._client.upsert(collection_name=self.collection, points=points)
            except Exception:  # noqa: BLE001, S110
                pass
        return len(vecs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qvec = self._embed(query)
        results: list[tuple[Chunk, float]] = []
        for sc in self._mem:
            score = sum(a * b for a, b in zip(qvec, sc.vector))
            results.append((sc.chunk, score))
        results.sort(key=lambda x: -x[1])
        if self._client and len(results) < top_k:
            try:
                hits = self._client.search(collection_name=self.collection, query_vector=qvec, limit=top_k)
                seen = {r[0].chunk_id for r in results}
                for h in hits:
                    payload = h.payload or {}
                    cid = payload.get("chunk_id")
                    if cid and cid in seen:
                        continue
                    # Reconstruct Chunk from payload
                    try:
                        chunk = Chunk(
                            chunk_id=payload.get("chunk_id", str(h.id)),
                            document_id=payload.get("document_id", ""),
                            content=payload.get("content", ""),
                            index=int(payload.get("index", 0)),
                            source=payload.get("source", ""),
                            section=payload.get("section") or None,
                            start=int(payload.get("start", 0)),
                            end=int(payload.get("end", 0)),
                        )
                    except Exception:  # noqa: BLE001, S112
                        continue
                    seen.add(chunk.chunk_id)
                    results.append((chunk, float(getattr(h, "score", 0.0))))
                results.sort(key=lambda x: -x[1])
            except Exception:  # noqa: BLE001, S110
                pass
        return results[:top_k]

    def snapshot(self, path: str) -> str:
        import json
        import pathlib

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "chunk_id": sc.chunk.chunk_id,
                "document_id": sc.chunk.document_id,
                "content": sc.chunk.content,
                "index": sc.chunk.index,
                "source": sc.chunk.source,
                "section": sc.chunk.section,
                "start": sc.chunk.start,
                "end": sc.chunk.end,
            }
            for sc in self._mem
        ]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def restore(self, path: str) -> int:
        import json
        import pathlib

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self._mem.clear()
        for d in data:
            c = Chunk(
                chunk_id=d["chunk_id"],
                document_id=d["document_id"],
                content=d["content"],
                index=int(d.get("index", 0)),
                source=d.get("source", ""),
                section=d.get("section"),
                start=int(d.get("start", 0)),
                end=int(d.get("end", 0)),
            )
            self._mem.append(StoredChunk(c, self._embed(c.content)))
        return len(self._mem)

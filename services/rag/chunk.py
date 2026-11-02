from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    index: int
    source: str
    section: str | None = None
    start: int = 0
    end: int = 0


def chunk_text(
    text: str,
    *,
    document_id: str,
    source: str,
    max_chars: int = 800,
    overlap: int = 120,
    section: str | None = None,
) -> list[Chunk]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", normalized) if s.strip()]
    chunks: list[Chunk] = []
    current = ""
    cursor = 0
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            start = cursor
            end = start + len(current)
            chunks.append(
                Chunk(
                    f"{document_id}:{len(chunks)}",
                    document_id,
                    current,
                    len(chunks),
                    source,
                    section,
                    start,
                    end,
                )
            )
            tail = current[-overlap:] if overlap else ""
            cursor = max(0, end - len(tail))
            current = f"{tail} {sentence}".strip()
        else:
            current = candidate
    if current:
        chunks.append(
            Chunk(
                f"{document_id}:{len(chunks)}",
                document_id,
                current,
                len(chunks),
                source,
                section,
                cursor,
                cursor + len(current),
            )
        )
    return chunks

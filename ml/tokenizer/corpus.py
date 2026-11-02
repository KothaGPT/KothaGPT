"""Corpus loading for tokenizer training.

Accepts a single text file, a JSON/JSONL (gzip) file, or a directory of
documents (reuses the Phase 1A ingestion formats).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

__all__ = ["iter_corpus", "load_corpus"]

_JSONL_EXTS = (".jsonl", ".jsonl.gz")


def load_corpus(path: str | Path) -> list[str]:
    """Return the list of documents under ``path`` as plain text."""
    return list(iter_corpus(path))


def iter_corpus(path: str | Path) -> Iterator[str]:
    """Yield documents under ``path`` one at a time (bounded memory).

    ``load_corpus`` is ``list(iter_corpus(path))``. Every yielded document
    comes from a record's ``text`` field; documents are yielded in stable
    order so the corpus digest is reproducible across streaming and eager use.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"corpus not found: {p}")
    if p.is_dir():
        from data.pipeline.io import iter_records

        texts = (record["text"] for record in iter_records(p))
    elif p.suffix == ".txt":
        texts = iter((p.read_text(encoding="utf-8"),))
    elif p.suffix in _JSONL_EXTS or p.name.endswith(".jsonl.gz"):
        texts = _iter_jsonl(p)
    elif p.suffix == ".json":
        texts = _iter_json(p)
    else:
        raise ValueError(f"unsupported corpus format: {p.suffix}")
    count = 0
    for text in texts:
        count += 1
        yield text
    if count == 0:
        raise ValueError(f"no documents found under {p}")


def _iter_jsonl(path: Path) -> Iterator[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError(f"{path}:{lineno}: missing 'text' field")
            yield record["text"]


def _iter_json(path: Path) -> Iterator[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a JSON array of objects")
    for record in data:
        if isinstance(record, dict):
            yield record["text"]

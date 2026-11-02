"""Input/output helpers: record iteration and JSONL shard writing."""

from __future__ import annotations

import gzip
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["iter_records", "write_shards"]

_SUPPORTED_EXTENSIONS = (
    ".txt",
    ".jsonl",
    ".jsonl.gz",
    ".json",
    ".html",
    ".htm",
    ".xml",
)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_records(source: Path) -> Iterator[dict[str, Any]]:
    """Yield ``{id, text, source}`` records from every file under ``source``.

    Supported formats:
      - ``*.txt``      one document per file
      - ``*.html``/``*.xml``  one document per file (markup stripped later)
      - ``*.jsonl``    one JSON object per line with a ``text`` field
      - ``*.json``     a JSON array of objects with a ``text`` field
    """
    files = sorted(
        p
        for p in source.rglob("*")
        if not p.name.startswith(".")
        and (p.name.endswith(".jsonl.gz") or p.suffix.lower() in _SUPPORTED_EXTENSIONS)
    )
    if not files:
        raise FileNotFoundError(f"No supported files found under {source}")
    for path in files:
        rel = str(path.relative_to(source))
        if path.suffix.lower() in (".txt", ".html", ".htm", ".xml"):
            text = _read_text_file(path)
            if text.strip():
                yield {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, rel)), "text": text, "source": rel}
        elif path.name.endswith(".jsonl.gz") or path.suffix.lower() == ".jsonl":
            if path.name.endswith(".jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as fh:  # type: ignore[assignment]
                    yield from _iter_jsonl(fh, rel)
            else:
                with path.open(encoding="utf-8") as fh:
                    yield from _iter_jsonl(fh, rel)
        elif path.suffix.lower() == ".json":
            data = json.loads(_read_text_file(path))
            if not isinstance(data, list):
                raise ValueError(f"{rel}: expected a JSON array of objects")
            for i, record in enumerate(data):
                if not isinstance(record, dict) or "text" not in record:
                    raise ValueError(f"{rel}:{i}: expected object with 'text' field")
                record.setdefault("source", rel)
                record.setdefault("id", str(uuid.uuid4()))
                yield record


def _iter_jsonl(fh, rel: str):
    for lineno, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "text" not in record:
            raise ValueError(f"{rel}:{lineno}: missing 'text' field")
        record.setdefault("source", rel)
        record.setdefault("id", str(uuid.uuid4()))
        yield record


def _open_shard(path: Path, gzip_output: bool):
    if gzip_output:
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def write_shards(
    records: Iterator[dict[str, Any]],
    out_dir: Path,
    *,
    shard_size: int = 100_000,
    gzip_output: bool = True,
) -> list[dict]:
    """Write records to ``out_dir`` as numbered JSONL shards.

    Returns a manifest list of ``{file, count}`` entries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict] = []
    shard_index = 0
    suffix = ".jsonl.gz" if gzip_output else ".jsonl"
    while True:
        shard_path = out_dir / f"shard-{shard_index:06d}{suffix}"
        count = 0
        with _open_shard(shard_path, gzip_output) as fh:
            while count < shard_size:
                try:
                    record = next(records)
                except StopIteration:
                    break
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            shard_path.unlink(missing_ok=True)
            break
        shards.append({"file": shard_path.name, "count": count})
        shard_index += 1
    return shards

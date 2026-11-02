"""Fetch the Bengali Wikipedia dump (text column) into data/raw/bn_wikipedia/.

Usage:
    python data/scripts/fetch_wikipedia.py [--config 20231101.bn] [--out-dir data/raw/bn_wikipedia]
    python data/scripts/fetch_wikipedia.py --limit-shards 1 --out-dir /tmp/opencode/wiki-smoke

Source: Wikimedia's `wikimedia/wikipedia` dataset on Hugging Face (parquet
shards with a pre-extracted `text` column). License: CC BY-SA 4.0 / GFDL
(Wikimedia terms; verify before production use).

Requires `pyarrow` (parquet reading); the rest is stdlib. This is a dev-time
fetch tool and is intentionally separate from the pure-stdlib pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

_DATASETS_SERVER = "https://datasets-server.huggingface.co/parquet"
_DATASET = "wikimedia/wikipedia"


def _parquet_urls(config: str, split: str, limit: int | None) -> list[dict]:
    url = f"{_DATASETS_SERVER}?dataset={_DATASET}&config={config}&split={split}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    files = data.get("parquet_files", [])
    if limit:
        files = files[:limit]
    return files


def _extract_text_parquet(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text"])
    return [str(v) for v in table.column("text").to_pylist()]


def _write_jsonl(texts: list[str], out_file: Path, source: str) -> None:
    import uuid

    with out_file.open("w", encoding="utf-8") as fh:
        for text in texts:
            record = {"id": str(uuid.uuid4()), "text": text, "source": source}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch(
    config: str, split: str, out_dir: Path, limit: int | None, records_per_file: int
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _parquet_urls(config, split, limit)
    if not files:
        raise ValueError(f"no parquet files found for {config}/{split}")
    print(f"Found {len(files)} parquet shard(s); extracting text column ...")
    written: list[str] = []
    for i, entry in enumerate(files, 1):
        url = entry["url"]
        shard = entry["url"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        print(f"[{i}/{len(files)}] {shard} ({entry.get('size', 0) // 1048576} MB) ...", flush=True)
        tmp = Path("/tmp") / f"wiki-{shard}.parquet"
        urllib.request.urlretrieve(url, tmp)
        texts = _extract_text_parquet(tmp)
        tmp.unlink(missing_ok=True)
        for start in range(0, len(texts), records_per_file):
            chunk = texts[start : start + records_per_file]
            out = out_dir / f"{shard}-{start // records_per_file:04d}.jsonl"
            _write_jsonl(chunk, out, source=f"bn_wikipedia/{shard}")
            written.append(out.name)
        print(f"  {len(texts)} records -> {out_dir}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="20231101.bn")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="data/raw/bn_wikipedia")
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--records-per-file", type=int, default=50_000)
    args = parser.parse_args(argv)

    try:
        written = fetch(
            args.config,
            args.split,
            Path(args.out_dir),
            args.limit_shards,
            args.records_per_file,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow":
            print(
                "fetch_wikipedia requires pyarrow: uv venv /tmp/corpus-venv && uv pip install pyarrow",
                file=sys.stderr,
            )
        raise
    print(f"Done. {len(written)} JSONL file(s) written under {args.out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WS-1/WS-2 — Pre-tokenize processed corpus shards into uint32 block shards.

Converts the processed corpus (``data/processed/<version>/``, gzipped JSONL
shards) into deterministic, memmap-friendly uint32 block shards so training
never re-tokenizes:

- ``data/tokenized/<corpus-version>-<tokenizer-digest>/{train,validation}/shard-<N>.bin``
  (contiguous uint32 values; each bin holds ``n_blocks * block_size`` ints)
  plus one sidecar ``shard-<N>.json`` per bin.
- ``MANIFEST.json`` listing splits, shards, token/block counts, the corpus
  version and tokenizer digest.

Packing follows ``ml/trainer/dataset.py``'s ``build_blocks`` semantics: records
are encoded in order and packed contiguously across document and shard
boundaries into ``block_size`` blocks; the final partial block of each split is
dropped. The train and validation splits are packed independently, and the
validation split is never mixed into training tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from array import array
from pathlib import Path
from typing import Any

from ml.tokenizer import load_tokenizer

__all__ = ["main", "tokenize_corpus"]

_BLOCK_SIZE = 4096


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _resolve_corpus(corpus: str | Path) -> Path:
    p = Path(corpus)
    for cand in (p, Path("data/processed") / p):
        if cand.is_dir() and (cand / "MANIFEST.json").is_file():
            return cand
    raise FileNotFoundError(f"no processed corpus at {p} (expected a dir with MANIFEST.json)")


def _read_shard_records(shard: Path):
    """Yield ``{text}`` records from a single .jsonl or .jsonl.gz shard."""
    import gzip
    import json

    opener = gzip.open if str(shard).endswith(".gz") else open
    with opener(shard, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError(f"{shard}:{lineno}: missing 'text' field")
            yield record


def _tokenize_split(
    split_dir: Path,
    tokenizer,
    block_size: int,
    out_split: Path,
) -> dict[str, Any]:
    out_split.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(split_dir.glob("*.jsonl.gz")) or sorted(split_dir.glob("*.jsonl"))
    if not shard_files:
        raise ValueError(f"no shards under {split_dir}")

    buffer = array("I")
    shards: list[dict[str, Any]] = []
    total_tokens = 0
    total_blocks = 0

    for idx, shard in enumerate(shard_files):
        input_tokens = 0
        for record in _read_shard_records(shard):
            ids = tokenizer.encode(record["text"])
            input_tokens += len(ids)
            buffer.extend(ids)
        # Flush complete blocks assembled while consuming this shard. A partial
        # block carries over into the next input shard (same semantics as
        # build_blocks over the whole split).
        n_full = len(buffer) // block_size
        if n_full:
            full = buffer[: n_full * block_size]
            bin_path = out_split / f"shard-{idx:06d}.bin"
            bin_path.write_bytes(full.tobytes())
            sidecar_path = out_split / f"shard-{idx:06d}.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "file": bin_path.name,
                        "index": idx,
                        "block_size": block_size,
                        "n_blocks": n_full,
                        "n_input_tokens": input_tokens,
                        "sha256": _sha256_file(bin_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            shards.append(
                {
                    "file": bin_path.name,
                    "index": idx,
                    "n_blocks": n_full,
                    "sha256": _sha256_file(bin_path),
                }
            )
            total_blocks += n_full
            del buffer[: n_full * block_size]
        total_tokens += input_tokens

    dropped_tokens = len(buffer)
    return {
        "shards": shards,
        "n_blocks": total_blocks,
        "n_tokens": total_tokens,
        "dropped_tokens": dropped_tokens,
    }


def _resolve_split_dir(corpus_root: Path, rel: str) -> Path:
    for cand in (corpus_root / rel, corpus_root.parent / rel):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"split dir not found: {rel} under {corpus_root}")


def tokenize_corpus(
    corpus: str | Path,
    tokenizer_path: str | Path,
    block_size: int,
    out_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    corpus_root = _resolve_corpus(corpus)
    manifest = json.loads((corpus_root / "MANIFEST.json").read_text(encoding="utf-8"))
    train_dir = _resolve_split_dir(corpus_root, manifest["files"]["train_dir"])
    val_dir = _resolve_split_dir(corpus_root, manifest["files"]["validation_dir"])
    tok_dir = Path(tokenizer_path)
    tok_file = tok_dir / "tokenizer.json"
    if not tok_file.is_file():
        tok_file = tok_dir
    tokenizer = load_tokenizer(tok_file)

    tokenizer_digest = _sha256_file(tok_file)
    corpus_version = manifest["version_id"]
    out_dir = Path(out_root) / f"{corpus_version}-{tokenizer_digest}-b{block_size}"
    if out_dir.exists() and not force:
        raise FileExistsError(f"{out_dir} already exists (re-run with --force to overwrite)")

    staging_dir = out_dir.with_name(out_dir.name + ".staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        train = _tokenize_split(train_dir, tokenizer, block_size, staging_dir / "train")
        validation = _tokenize_split(val_dir, tokenizer, block_size, staging_dir / "validation")

        for name, result in (("train", train), ("validation", validation)):
            if result["n_blocks"] == 0:
                validation_error = ValueError(
                    f"{name} split produced 0 complete blocks at block_size={block_size} "
                    f"({result['n_tokens']} tokens available); use a larger corpus or a "
                    f"smaller --block-size"
                )
                try:
                    shutil.rmtree(staging_dir)
                except Exception as cleanup_err:  # noqa: BLE001
                    raise RuntimeError(
                        f"failed to cleanup staging {staging_dir}: {cleanup_err}"
                    ) from validation_error
                if staging_dir.exists():
                    raise RuntimeError(f"staging {staging_dir} still exists after cleanup")
                raise validation_error

        top = {
            "format": "uint32 blocks",
            "block_size": block_size,
            "corpus_version": corpus_version,
            "corpus_dir": str(corpus_root),
            "tokenizer_digest": tokenizer_digest,
            "tokenizer_path": str(tok_dir),
            "splits": {"train": train, "validation": validation},
            "total_blocks": train["n_blocks"] + validation["n_blocks"],
            "total_tokens": train["n_tokens"] + validation["n_tokens"],
        }
        (staging_dir / "MANIFEST.json").write_text(
            json.dumps(top, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except Exception:  # noqa: BLE001, S110
                pass
        raise

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging_dir.rename(out_dir)
    (Path(out_root) / "CURRENT").write_text(out_dir.name + "\n", encoding="utf-8")
    return top


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default=None,
        help="processed corpus dir or version id (default: data/processed/CURRENT)",
    )
    parser.add_argument("--tokenizer", default="ml/tokenizer/artifacts/best", help="tokenizer dir")
    parser.add_argument("--block-size", type=int, default=_BLOCK_SIZE, help="token block size")
    parser.add_argument("--out", default="data/tokenized", help="output root")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args(argv)

    corpus = args.corpus
    if corpus is None:
        current = Path("data/processed/CURRENT")
        if current.is_file():
            corpus = current.read_text(encoding="utf-8").strip()
        else:
            raise SystemExit("no --corpus given and data/processed/CURRENT is missing")

    result = tokenize_corpus(corpus, args.tokenizer, args.block_size, args.out, force=args.force)
    print(
        f"tokenized {result['total_tokens']:,} tokens into "
        f"{result['total_blocks']:,} blocks (block_size={result['block_size']:,})",
        flush=True,
    )
    for split, data in result["splits"].items():
        print(
            f"  {split}: {data['n_blocks']:,} blocks / {data['n_tokens']:,} tokens "
            f"(dropped {data['dropped_tokens']:,})",
            flush=True,
        )
    print(
        f"wrote MANIFEST to {Path(args.out) / (result['corpus_version'] + '-' + result['tokenizer_digest'] + '-b' + str(result['block_size']))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

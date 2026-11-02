"""Deduplication: exact (sha256) and near-duplicate (MinHash + LSH).

Exact duplicates are removed with a content hash set. Near-duplicates are
detected with word-shingle MinHash signatures bucketed by banded LSH, which
keeps memory bounded while remaining robust to small edits.

Scaling helpers:

- :class:`BloomFilter` bounds the exact-dedup seen set to a fixed number of
  bits at a configurable false-positive rate (opt-in).
- :class:`ExactDedupState` persists the exact seen-set to disk so subsequent
  runs of other corpora are deduplicated against the whole dataset
  (cross-source dedup).
- :func:`near_duplicate_groups_sharded` spills band buckets and signatures to
  disk so near-dedup memory stays bounded by the largest bucket, not the whole
  corpus.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

__all__ = [
    "BloomFilter",
    "ExactDedupState",
    "deduplicate",
    "deduplicate_with_stats",
    "exact_duplicate_keys",
    "minhash",
    "near_duplicate_groups",
    "near_duplicate_groups_sharded",
    "text_hash",
]

_RECORD_TEXT_KEY = "text"

_SIG_INT_BYTES = 8


def text_hash(text: str) -> str:
    """Content hash used for exact dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_duplicate_keys(records: Iterable[str], *, seen: set[str] | None = None):
    """Yield ``(text, is_new)`` for each input text, tracking exact dupes."""
    if seen is None:
        seen = set()
    for text in records:
        digest = text_hash(text)
        is_new = digest not in seen
        if is_new:
            seen.add(digest)
        yield text, is_new


class BloomFilter:
    """Fixed-memory membership filter with a configurable false-positive rate.

    Implemented as a ``bytearray`` bit array with double hashing (sha256 +
    sha512 seeds). Only ``add``/``__contains__`` (digest ``str``) are needed to
    act as an exact-dedup seen store; false positives drop a real duplicate.
    """

    def __init__(self, capacity: int, fp_rate: float = 0.01):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 < fp_rate < 1.0:
            raise ValueError("fp_rate must be in (0, 1)")
        ln2 = math.log(2)
        bits = max(8, math.ceil(-capacity * math.log(fp_rate) / (ln2 * ln2)))
        self._m = bits
        self._k = max(1, round((bits / capacity) * ln2))
        self._bits = bytearray((bits + 7) // 8)
        self._count = 0

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    @property
    def count(self) -> int:
        return self._count

    def _hashes(self, key: str) -> list[int]:
        h1 = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        h2 = int.from_bytes(hashlib.sha512(key.encode()).digest()[:8], "big")
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, key: str) -> None:
        for pos in self._hashes(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def __contains__(self, key: str) -> bool:
        return all(self._bits[pos >> 3] & (1 << (pos & 7)) for pos in self._hashes(key))

    has = __contains__


class ExactDedupState:
    """Persistent exact seen-set (one sha256 digest per line).

    Lets a new corpus be deduplicated against everything seen in earlier runs,
    preventing one source from leaking duplicate text into another. Mark
    digests as you go and call :meth:`save` when the pipeline finishes.
    """

    def __init__(self, path: str | os.PathLike):
        self._path = Path(path)
        self._seen: set[str] = set()
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._seen.add(line)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return len(self._seen)

    def __contains__(self, digest: str) -> bool:
        return digest in self._seen

    def add(self, digest: str) -> None:
        self._seen.add(digest)

    def is_new(self, text: str) -> bool:
        return text_hash(text) not in self._seen

    def mark(self, text: str) -> None:
        self._seen.add(text_hash(text))

    def save(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            "\n".join(sorted(self._seen)) + ("\n" if self._seen else ""),
            encoding="utf-8",
        )
        return self._path


def _hash_bytes(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _shingles(text: str, k: int = 5) -> list[str]:
    words = text.split()
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def minhash(
    text: str,
    *,
    num_hashes: int = 128,
    shingle_size: int = 5,
) -> list[int]:
    """Compute a MinHash signature for ``text`` (pure python, deterministic)."""
    sig = [2**63 - 1] * num_hashes
    for shingle in _shingles(text, shingle_size):
        for i in range(num_hashes):
            value = _hash_bytes(f"{i}\x00{shingle}".encode())
            sig[i] = min(sig[i], value)
    return sig


def _signature_bytes(sig: list[int]) -> bytes:
    return b"".join(v.to_bytes(_SIG_INT_BYTES, "big") for v in sig)


def _jaccard_estimate(a: list[int], b: list[int]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _union_find_groups(members: Iterable[str]) -> tuple[dict[str, str], list[set[str]]]:
    """Union-find forest over ``members``; returns (parent, groups)."""
    parent = {key: key for key in members}
    return parent, None  # groups are computed via _collect_groups


def _collect_groups(parent: dict[str, str], members: Iterable[str]) -> list[set[str]]:
    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    by_root: dict[str, set[str]] = defaultdict(set)
    for key in members:
        by_root[find(key)].add(key)
    return [m for m in by_root.values() if len(m) > 1]


def near_duplicate_groups(
    signatures: dict[str, list[int]],
    *,
    num_bands: int = 32,
    rows_per_band: int = 4,
    threshold: float = 0.8,
) -> list[set[str]]:
    """Return connected groups of near-duplicate keys via banded LSH.

    A pair whose signatures collide in any band is treated as a candidate and
    confirmed with an exact Jaccard estimate. Bands x rows must equal the
    signature length.
    """
    expected = num_bands * rows_per_band
    for sig in signatures.values():
        if len(sig) != expected:
            raise ValueError(f"signature length {len(sig)} != bands*rows={expected}")

    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, sig in signatures.items():
        for b in range(num_bands):
            band = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
            buckets[band].append(key)

    candidates: set[frozenset[str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add(frozenset((members[i], members[j])))

    parent, _ = _union_find_groups(signatures)
    for pair in candidates:
        ka, kb = tuple(pair)
        if _jaccard_estimate(signatures[ka], signatures[kb]) >= threshold:
            ra = find_root(parent, ka)
            rb = find_root(parent, kb)
            if ra != rb:
                parent[ra] = rb

    return _collect_groups(parent, signatures)


def find_root(parent: dict[str, str], key: str) -> str:
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key


def near_duplicate_groups_sharded(
    records: Iterable[dict[str, Any]],
    *,
    text_key: str = _RECORD_TEXT_KEY,
    num_hashes: int = 128,
    shingle_size: int = 5,
    num_bands: int = 32,
    rows_per_band: int = 4,
    threshold: float = 0.8,
    tmp_dir: str | os.PathLike | None = None,
) -> list[set[str]]:
    """Disk-spilling banded LSH; memory bounded by the largest bucket.

    Signatures are appended to one fixed-size binary file and band buckets to
    per-band text files under a temp dir. Candidate pairs inside each bucket
    are confirmed against the on-disk signatures. Returns the same shape as
    :func:`near_duplicate_groups`.
    """
    expected = num_hashes
    if expected != num_bands * rows_per_band:
        raise ValueError(f"num_hashes {expected} != bands*rows={num_bands * rows_per_band}")

    work = Path(tempfile.mkdtemp(prefix="kothagpt-dedup-", dir=str(tmp_dir) if tmp_dir else None))
    try:
        sigs_path = work / "signatures.bin"
        key_to_idx: dict[str, int] = {}
        with ExitStack() as stack:
            band_fhs = [
                stack.enter_context(open(work / f"band_{b:03d}.txt", "a", encoding="utf-8"))
                for b in range(num_bands)
            ]
            with open(sigs_path, "wb") as sigs_fh:
                for idx, record in enumerate(records):
                    key = text_hash(record[text_key])
                    key_to_idx[key] = idx
                    sig = minhash(
                        record[text_key], num_hashes=num_hashes, shingle_size=shingle_size
                    )
                    sigs_fh.write(_signature_bytes(sig))
                    for b in range(num_bands):
                        band_fhs[b].write(key + "\n")

        record_size = expected * _SIG_INT_BYTES

        def read_sig(idx: int) -> list[int]:
            with open(sigs_path, "rb") as fh:
                fh.seek(idx * record_size)
                data = fh.read(record_size)
            return [
                int.from_bytes(data[i : i + _SIG_INT_BYTES], "big")
                for i in range(0, record_size, _SIG_INT_BYTES)
            ]

        parent = {key: key for key in key_to_idx}
        for band_path in sorted(work.glob("band_*.txt")):
            members = [
                line.strip()
                for line in band_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ka, kb = members[i], members[j]
                    if (
                        _jaccard_estimate(read_sig(key_to_idx[ka]), read_sig(key_to_idx[kb]))
                        >= threshold
                    ):
                        ra = find_root(parent, ka)
                        rb = find_root(parent, kb)
                        if ra != rb:
                            parent[ra] = rb

        return _collect_groups(parent, key_to_idx)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def deduplicate_with_stats(
    records: Iterable[dict[str, Any]],
    *,
    exact: bool = True,
    near: bool = False,
    threshold: float = 0.8,
    num_hashes: int = 128,
    shingle_size: int = 5,
    num_bands: int = 32,
    rows_per_band: int = 4,
    text_key: str = _RECORD_TEXT_KEY,
    log=None,
    exact_store: set[str] | BloomFilter | ExactDedupState | None = None,
    near_sharded: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove duplicate records; returns ``(kept, counts)``.

    ``counts`` reports ``input``, ``removed_exact`` and ``removed_near``.
    Exactly equal normalized texts are always removed when ``exact``. When
    ``near`` is enabled, records are grouped by banded MinHash LSH and all but
    the first (by input order) record per group is dropped. ``exact_store`` may
    be a ``set``, :class:`BloomFilter` or :class:`ExactDedupState` (anything
    supporting ``__contains__``/``add`` over sha256 digests).
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] | BloomFilter | ExactDedupState = (
        exact_store if exact_store is not None else set()
    )
    removed_exact = 0
    for record in records:
        text = record[text_key]
        if exact:
            digest = text_hash(text)
            if digest in seen:
                removed_exact += 1
                continue
            seen.add(digest)
        kept.append(record)

    removed_near = 0
    if near and len(kept) > 1:
        if near_sharded:
            groups = near_duplicate_groups_sharded(
                kept,
                text_key=text_key,
                num_hashes=num_hashes,
                shingle_size=shingle_size,
                num_bands=num_bands,
                rows_per_band=rows_per_band,
                threshold=threshold,
            )
        else:
            signatures = {
                text_hash(record[text_key]): minhash(
                    record[text_key],
                    num_hashes=num_hashes,
                    shingle_size=shingle_size,
                )
                for record in kept
            }
            groups = near_duplicate_groups(
                signatures,
                num_bands=num_bands,
                rows_per_band=rows_per_band,
                threshold=threshold,
            )
        drop: set[str] = set()
        for group in groups:
            for member in sorted(group)[1:]:
                drop.add(member)
        removed_near = len(drop)
        if removed_near:
            kept = [r for r in kept if text_hash(r[text_key]) not in drop]

    counts = {
        "input": removed_exact + len(kept),
        "removed_exact": removed_exact,
        "removed_near": removed_near,
    }
    return kept, counts


def deduplicate(
    records: Iterable[dict[str, Any]],
    *,
    exact: bool = True,
    near: bool = False,
    threshold: float = 0.8,
    num_hashes: int = 128,
    shingle_size: int = 5,
    num_bands: int = 32,
    rows_per_band: int = 4,
    text_key: str = _RECORD_TEXT_KEY,
    log=None,
) -> list[dict[str, Any]]:
    """Remove duplicate records; returns the deduplicated list.

    Exactly equal normalized texts are always removed when ``exact``. When
    ``near`` is enabled, records are grouped by banded MinHash LSH and all but
    the first (by input order) record per group is dropped.
    """
    kept, _ = deduplicate_with_stats(
        records,
        exact=exact,
        near=near,
        threshold=threshold,
        num_hashes=num_hashes,
        shingle_size=shingle_size,
        num_bands=num_bands,
        rows_per_band=rows_per_band,
        text_key=text_key,
        log=log,
    )
    return kept

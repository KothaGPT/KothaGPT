"""Streaming corpus dataset for causal-LM pretraining.

Records are tokenized in deterministic order and packed contiguously into
``block_size`` blocks (standard LM packing across document boundaries). The
final partial block is dropped to avoid padding. Blocks are materialized as a
single ``LongTensor``; for very large corpora, build per shard.

Two datasets are provided:

- :class:`CausalLMDataset` — the eager, in-RAM dataset over ``build_blocks``.
  Good for small corpora and smoke runs.
- :class:`ShardedMemmapDataset` — the WS-3 memmap-backed dataset over
  pre-tokenized shards (``ml/tokenize_shards.py`` output). Each rank memmaps
  only the shards assigned to it (disjoint reads across ranks), keeps RSS flat,
  and exposes the same ``(input_ids, labels)`` causal-shift API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.tokenizer import BaseTokenizer
from ml.tokenizer.corpus import load_corpus

__all__ = ["CausalLMDataset", "ShardedMemmapDataset", "build_blocks"]


def build_blocks(
    corpus: str | Path,
    tokenizer: BaseTokenizer,
    block_size: int,
) -> torch.Tensor:
    """Tokenize the corpus and pack it into ``[N, block_size]`` token blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    for text in load_corpus(corpus):
        ids = tokenizer.encode(text)
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= block_size:
            blocks.append(torch.tensor(buffer[:block_size], dtype=torch.long))
            del buffer[:block_size]
    if not blocks:
        raise ValueError(f"corpus too small for block_size={block_size}: {corpus}")
    return torch.stack(blocks)


class CausalLMDataset(Dataset):
    """Returns ``(input_ids, labels)`` with a one-token causal shift."""

    def __init__(self, blocks: torch.Tensor) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.blocks[index]
        return block[:-1], block[1:]


class ShardedMemmapDataset(Dataset):
    """Memmap-backed dataset over pre-tokenized ``.bin`` shards.

    Reads the ``MANIFEST.json`` written by ``ml/tokenize_shards.py`` and maps
    each assigned shard with :func:`numpy.memmap` (read-only, no copy). Blocks
    are returned with the same ``(input_ids, labels)`` causal shift as
    :class:`CausalLMDataset`, so the training loop is unchanged.

    For distributed runs pass ``rank``/``world_size`` to assign each rank a
    disjoint set of shard files (round-robin), so ranks never read the same
    shard and I/O does not duplicate across the cluster.
    """

    def __init__(
        self,
        tokenized_dir: str | Path,
        *,
        split: str = "train",
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        tokenized_dir = Path(tokenized_dir)
        manifest_path = tokenized_dir / "MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{tokenized_dir} is not a tokenized corpus (no MANIFEST.json); "
                "run `python -m ml.tokenize_shards` first"
            )
        manifest = _load_json(manifest_path)
        if manifest.get("format") != "uint32 blocks":
            raise ValueError(f"unsupported tokenized format in {manifest_path}")
        self.block_size = int(manifest["block_size"])
        self.split = split

        if "splits" in manifest:
            split_entry = manifest["splits"].get(split)
            if split_entry is None:
                raise ValueError(
                    f"split {split!r} not in {manifest_path}: {sorted(manifest['splits'])}"
                )
            shards = split_entry.get("shards") or []
            base_dir = tokenized_dir / split
        else:
            shards = manifest.get("shards") or []
            base_dir = tokenized_dir
        if not shards:
            raise ValueError(f"no shards for split {split!r} in {manifest_path}")

        if world_size and rank is not None:
            if not 0 <= rank < world_size:
                raise ValueError(f"rank {rank} out of range [0, {world_size})")
            shards = [s for i, s in enumerate(shards) if i % world_size == rank]
            if not shards:
                raise ValueError(f"rank {rank}/{world_size} has no shards")

        self._files: list[Path] = []
        self._offsets: list[int] = []  # first block index of each shard
        self._maps: list[np.ndarray | None] = []
        total = 0
        for entry in shards:
            self._files.append(base_dir / entry["file"])
            self._offsets.append(total)
            self._maps.append(None)
            total += int(entry["n_blocks"])
        self._n_blocks = total
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int | None:
        return self._rank

    @property
    def world_size(self) -> int | None:
        return self._world_size

    def __len__(self) -> int:
        return self._n_blocks

    def _shard_for(self, index: int) -> tuple[np.ndarray, int]:
        # Binary search for the shard containing `index` (offsets are sorted).
        lo, hi = 0, len(self._offsets)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._offsets[mid] <= index:
                lo = mid
            else:
                hi = mid
        arr = self._maps[lo]
        if arr is None:
            arr = np.memmap(
                self._files[lo],
                dtype=np.uint32,
                mode="r",
                shape=(
                    self._offsets[lo + 1] - self._offsets[lo]
                    if lo + 1 < len(self._offsets)
                    else self._n_blocks - self._offsets[lo],
                    self.block_size,
                ),
            )
            self._maps[lo] = arr
        return arr, index - self._offsets[lo]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self._n_blocks:
            raise IndexError(f"index {index} out of range [0, {self._n_blocks})")
        arr, local = self._shard_for(index)
        block = torch.from_numpy(arr[local].astype(np.int64))
        return block[:-1], block[1:]

    def close(self) -> None:
        """Release memmaps (np.memmap has no explicit close; drop references)."""
        self._maps = [None] * len(self._maps)


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

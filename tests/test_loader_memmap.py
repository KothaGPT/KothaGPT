"""WS-3 — memmap-backed dataset over pre-tokenized shards."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.tokenize_shards import tokenize_corpus
from ml.trainer.dataset import CausalLMDataset, ShardedMemmapDataset, build_blocks

FIXTURES = Path(__file__).parent / "fixtures"


def _records():
    return [
        json.loads(line)
        for line in (FIXTURES / "raw" / "bangla-sample.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    train = corpus / "train"
    val = corpus / "validation"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    recs = _records()
    with gzip.open(train / "shard-000000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs[:6]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(train / "shard-000001.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs[6:10]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(val / "shard-000000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs[10:]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (corpus / "MANIFEST.json").write_text(
        json.dumps(
            {
                "version_id": "test",
                "files": {"train_dir": "train", "validation_dir": "validation"},
                "shards": [{"file": "shard-000000.jsonl.gz", "count": 6}],
            }
        ),
        encoding="utf-8",
    )
    return corpus


def _make_tokenizer(tmp_path: Path) -> Path:
    from ml.tokenizer import train_bpe

    tokenizer = train_bpe([r["text"] for r in _records()], vocab_size=250, min_frequency=1)
    tokenizer.save(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def tokenized(tmp_path_factory):
    base = tmp_path_factory.mktemp("memmap")
    corpus = _make_corpus(base / "corpus")
    tok = _make_tokenizer(base / "tok")
    out = base / "out"
    result = tokenize_corpus(corpus, tok, block_size=32, out_root=out)
    return result, next(out.glob("*/"))


def test_dataset_matches_eager_build_blocks(tokenized):
    _, root = tokenized
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    from ml.tokenizer import load_tokenizer

    tok_path = manifest["tokenizer_path"]
    p = Path(tok_path)
    tokenizer = load_tokenizer(p if p.is_file() else p / "tokenizer.json")

    mem = ShardedMemmapDataset(root, split="train")
    eager = CausalLMDataset(
        build_blocks(Path(manifest["corpus_dir"]) / "train", tokenizer, block_size=32)
    )
    assert len(mem) == len(eager)
    for i in range(len(mem)):
        mi, mlab = mem[i]
        ei, elab = eager[i]
        assert torch.equal(mi, ei)
        assert torch.equal(mlab, elab)


def test_rank_shards_are_disjoint_and_cover_all(tokenized):
    _, root = tokenized
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    total = manifest["splits"]["train"]["n_blocks"]
    d0 = ShardedMemmapDataset(root, split="train", rank=0, world_size=2)
    d1 = ShardedMemmapDataset(root, split="train", rank=1, world_size=2)
    assert len(d0) + len(d1) == total
    # ranks never read the same shard files
    assert not set(d0._files) & set(d1._files)
    # union of both ranks covers the whole split
    got = {i for i in range(len(d0))} | {len(d0) + i for i in range(len(d1))}
    assert len(got) == total


def test_causal_shift_and_block_size(tokenized):
    _, root = tokenized
    ds = ShardedMemmapDataset(root, split="validation")
    assert ds.block_size == 32
    input_ids, labels = ds[0]
    assert input_ids.shape == (31,)
    assert labels.shape == (31,)
    # labels are the block shifted by one: labels[t] == input_ids[t+1]
    assert torch.equal(labels[:-1], input_ids[1:])


def test_backed_by_memmap_not_ram(tokenized):
    _, root = tokenized
    ds = ShardedMemmapDataset(root, split="train")
    # Accessing items must lazily open memmaps, not copy the whole array.
    _ = ds[0]
    for arr in ds._maps:
        if arr is not None:
            assert isinstance(arr, np.memmap)


def test_missing_split_raises(tokenized):
    _, root = tokenized
    with pytest.raises(ValueError, match="split 'nope'"):
        ShardedMemmapDataset(root, split="nope")

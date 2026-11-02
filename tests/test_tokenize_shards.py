"""WS-1/WS-2 — tokenize_shards pipeline tests."""

import gzip
import json
from pathlib import Path

import pytest

from ml.tokenize_shards import tokenize_corpus

FIXTURES = Path(__file__).parent / "fixtures"


def _make_corpus(tmp_path: Path) -> Path:
    """Build a tiny processed-style corpus (train + validation, gz shards)."""
    records = []
    for line in (FIXTURES / "raw" / "bangla-sample.jsonl").read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    corpus = tmp_path / "corpus"
    train = corpus / "train"
    val = corpus / "validation"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    with gzip.open(train / "shard-000000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records[:8]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with gzip.open(train / "shard-000001.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records[8:10]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with gzip.open(val / "shard-000000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records[10:]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (corpus / "MANIFEST.json").write_text(
        json.dumps(
            {
                "version_id": "test-corpus",
                "version_label": "test",
                "files": {
                    "train_dir": "train",
                    "validation_dir": "validation",
                },
                "shards": [{"file": "shard-000000.jsonl.gz", "count": 8}],
            }
        ),
        encoding="utf-8",
    )
    return corpus


def _make_tokenizer(tmp_path: Path) -> Path:
    from ml.tokenizer import train_bpe

    texts = []
    for line in (FIXTURES / "raw" / "bangla-sample.jsonl").read_text(encoding="utf-8").splitlines():
        texts.append(json.loads(line)["text"])
    tokenizer = train_bpe(texts, vocab_size=250, min_frequency=1)
    tokenizer.save(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def tokenized(tmp_path_factory):
    base = tmp_path_factory.mktemp("tok")
    corpus = _make_corpus(base / "make-corpus")
    tok = _make_tokenizer(base / "make-tok")
    out = base / "out"
    result = tokenize_corpus(corpus, tok, block_size=32, out_root=out)
    manifest_path = next(out.glob("*/MANIFEST.json"))
    return result, manifest_path


def _read_blocks(bin_path: Path, block_size: int) -> list[list[int]]:
    data = bin_path.read_bytes()
    return [
        [int.from_bytes(data[i : i + 4], "little") for i in range(off, off + block_size * 4, 4)]
        for off in range(0, len(data), block_size * 4)
    ]


def _load_tokenizer(tokenizer_path):
    from ml.tokenizer import load_tokenizer

    p = Path(tokenizer_path)
    if not p.is_file():
        p = p / "tokenizer.json"
    return load_tokenizer(p)


def test_manifest_and_splits(tokenized):
    result, manifest_path = tokenized
    assert result["block_size"] == 32
    assert set(result["splits"]) == {"train", "validation"}
    assert result["total_blocks"] == (
        result["splits"]["train"]["n_blocks"] + result["splits"]["validation"]["n_blocks"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["block_size"] == 32
    assert manifest["tokenizer_digest"] == result["tokenizer_digest"]
    assert manifest["corpus_version"] == "test-corpus"


def test_blocks_are_full_and_contiguous(tokenized):
    _, manifest_path = tokenized
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_dir = manifest_path.parent / "train"
    for shard in manifest["splits"]["train"]["shards"]:
        blocks = _read_blocks(split_dir / shard["file"], manifest["block_size"])
        assert all(len(b) == manifest["block_size"] for b in blocks)
        assert len(blocks) == shard["n_blocks"]
        raw = (split_dir / shard["file"]).read_bytes()
        assert raw == b"".join(b"".join(v.to_bytes(4, "little") for v in b) for b in blocks)


def test_idempotent(tokenized, tmp_path):
    base = tmp_path / "base"
    corpus = _make_corpus(base / "corpus")
    tok = _make_tokenizer(base / "tok")
    out2 = base / "out2"
    result2 = tokenize_corpus(corpus, tok, block_size=32, out_root=out2)
    _, manifest_path1 = tokenized
    manifest_path2 = next(out2.glob("*/MANIFEST.json"))
    for rel in sorted(
        p.relative_to(manifest_path1.parent) for p in manifest_path1.parent.rglob("*.bin")
    ):
        assert (manifest_path1.parent / rel).read_bytes() == (
            manifest_path2.parent / rel
        ).read_bytes()
    m1 = json.loads(manifest_path1.read_text(encoding="utf-8"))
    m2 = json.loads(manifest_path2.read_text(encoding="utf-8"))
    assert m1["splits"] == m2["splits"]
    assert result2["tokenizer_digest"] == m2["tokenizer_digest"]


def test_rejects_non_positive_block_size(tmp_path):
    base = tmp_path / "base"
    corpus = _make_corpus(base / "corpus")
    tok = _make_tokenizer(base / "tok")
    for bad in (0, -1, -32):
        with pytest.raises(ValueError):
            tokenize_corpus(corpus, tok, block_size=bad, out_root=base / f"out{bad}")
    assert not any((base / "corpus").rglob("out*"))


def test_rejects_block_size_larger_than_corpus(tmp_path):
    """A split that cannot fill a single block must fail fast, not write an
    empty artifact (which only surfaces later as a confusing trainer error)."""
    base = tmp_path / "base"
    corpus = _make_corpus(base / "corpus")
    tok = _make_tokenizer(base / "tok")
    with pytest.raises(ValueError, match="0 complete blocks"):
        tokenize_corpus(corpus, tok, block_size=1_000_000, out_root=base / "out")
    # No partial artifact or CURRENT pointer is left behind.
    assert not any((base / "out").iterdir())


def test_matches_trainer_build_blocks(tokenized):
    """Blocks must equal ml/trainer/dataset.build_blocks over the train split."""
    pytest.importorskip("torch")
    _, manifest_path = tokenized
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from ml.trainer.dataset import build_blocks

    tokenizer = _load_tokenizer(manifest["tokenizer_path"])
    expected = build_blocks(
        Path(manifest["corpus_dir"]) / "train",
        tokenizer,
        block_size=manifest["block_size"],
    ).tolist()
    got = []
    for shard in manifest["splits"]["train"]["shards"]:
        got.extend(
            _read_blocks(manifest_path.parent / "train" / shard["file"], manifest["block_size"])
        )
    assert got == expected


def test_token_count_matches_encoding(tokenized):
    result, manifest_path = tokenized
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tokenizer = _load_tokenizer(manifest["tokenizer_path"])
    records = [
        json.loads(line)
        for line in (FIXTURES / "raw" / "bangla-sample.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    train_total = sum(len(tokenizer.encode(r["text"])) for r in records[:10])
    val_total = sum(len(tokenizer.encode(r["text"])) for r in records[10:])
    assert result["splits"]["train"]["n_tokens"] == train_total
    assert result["splits"]["validation"]["n_tokens"] == val_total
    assert result["splits"]["train"]["dropped_tokens"] < result["block_size"]
    assert result["splits"]["validation"]["dropped_tokens"] < result["block_size"]

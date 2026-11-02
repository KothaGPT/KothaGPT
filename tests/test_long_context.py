"""WS-11 — Long-context training tests (RoPE scaling + extension)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from ml.models import KothaGPT, ModelConfig, RotaryEmbedding, load_config
from ml.tokenizer import train_bpe
from ml.trainer import CausalLMDataset, build_blocks

_FILLER_SENT = "এটি একটি সাধারণ উদাহরণ বাক্য, সংখ্যা {i}। বাংলা ভাষা সমৃদ্ধ ইন্দো-আর্য ভাষা। "
_REFERENCE = "বাংলা ভাষা সমৃদ্ধ ইন্দো-আর্য ভাষা।"


def _filler_tokens(tokenizer, count: int) -> str:
    sent = _FILLER_SENT.format(i=1)
    toks = tokenizer.encode(sent)
    reps = count // len(toks) + 1
    return tokenizer.decode(tokenizer.encode(sent * reps)[:count])


@pytest.fixture(scope="module")
def long_ctx(tmp_path_factory: pytest.TempPathFactory):
    """A tokenizer + tiny model trained at context 32 on synthetic Bangla."""
    from ml.tokenizer import load_tokenizer

    tmp = tmp_path_factory.mktemp("long")
    corpus = tmp / "corpus.jsonl"
    docs = []
    for d in range(120):
        text = "".join(_FILLER_SENT.format(i=(i + d) % 50) for i in range(30))
        docs.append(json.dumps({"text": text}, ensure_ascii=False))
    corpus.write_text("\n".join(docs) + "\n", encoding="utf-8")
    texts = [json.loads(line)["text"] for line in docs]
    tok_dir = tmp / "tok"
    tokenizer = train_bpe(texts, vocab_size=500, min_frequency=1)
    tokenizer.save(tok_dir)
    tokenizer = load_tokenizer(tok_dir / "tokenizer.json")

    cfg = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
    )
    model = KothaGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    blocks = build_blocks(corpus, tokenizer, block_size=32)
    ds = CausalLMDataset(blocks)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    torch.manual_seed(0)
    model.train()
    for _ in range(150):
        x, y = next(iter(loader))
        loss = model(x, labels=y)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "cfg": cfg}


def _ppl(model: KothaGPT, tokenizer, prefix: str, reference: str) -> float:
    prefix_ids = tokenizer.encode(prefix)
    ref_ids = tokenizer.encode(reference)
    ids = torch.tensor([prefix_ids + ref_ids])
    labels = torch.full_like(ids, -100)
    labels[0, len(prefix_ids) - 1 : len(prefix_ids) - 1 + len(ref_ids)] = torch.tensor(ref_ids)
    loss = model(ids, labels=labels)["loss"].item()
    return math.exp(min(loss, 20))


def test_long_config_round_trips() -> None:
    cfg = load_config("ml/configs/long.yaml")
    assert cfg.model.max_position_embeddings == 16384
    assert cfg.model.rope_theta == 500000.0
    assert cfg.model.rope_scaling == "linear"
    assert cfg.model.rope_scaling_factor == 4.0


def test_ntk_scaling_widens_frequencies() -> None:
    plain = RotaryEmbedding(16, 64, theta=10000.0)
    ntk = RotaryEmbedding(16, 64, theta=10000.0, scaling="ntk", scaling_factor=4.0)
    assert not torch.allclose(plain.inv_freq, ntk.inv_freq)
    theta_eff = 10000.0 * 4.0 ** (16 / 14)
    expected = 1.0 / (theta_eff ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    assert torch.allclose(ntk.inv_freq, expected)


def test_linear_scaling_compresses_positions() -> None:
    emb = RotaryEmbedding(16, 64, theta=10000.0, scaling="linear", scaling_factor=4.0)
    cos_1, _ = emb.cos_cached[0], emb.cos_cached
    assert cos_1 is not None
    assert emb.cos_cached.shape[0] == 64
    single = RotaryEmbedding(16, 64, theta=10000.0)
    # position 4 under linear(4x) == position 1 unscaled
    assert torch.allclose(emb.cos_cached[4], single.cos_cached[1])


def test_extend_context_length_preserves_weights() -> None:
    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    model = KothaGPT(cfg)
    before = {n: p.clone() for n, p in model.named_parameters()}
    model.extend_context_length(
        64, rope_theta=500000.0, rope_scaling="ntk", rope_scaling_factor=4.0
    )
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p), n
    assert model.config.max_position_embeddings == 64
    assert model.config.rope_scaling == "ntk"
    assert model.rotary_emb.cos_cached.shape[0] == 64
    x = torch.randint(0, 128, (1, 64))
    out = model(x)
    assert out["logits"].shape == (1, 64, 128)


def test_extend_rejects_shrink() -> None:
    model = KothaGPT(
        ModelConfig(
            vocab_size=128, hidden_size=32, num_layers=2, num_heads=4, max_position_embeddings=16
        )
    )
    with pytest.raises(ValueError):
        model.extend_context_length(16)


def test_extension_keeps_ppl_stable_within_5pct(long_ctx) -> None:
    """WS-11 metric: extending a ctx-32 model to 64 keeps reference ppl < 5% worse."""
    model = long_ctx["model"]
    tokenizer = long_ctx["tokenizer"]
    baseline_prefix = _filler_tokens(tokenizer, 16)
    assert len(tokenizer.encode(baseline_prefix)) == 16
    assert len(tokenizer.encode(baseline_prefix)) + len(tokenizer.encode(_REFERENCE)) <= 32
    baseline_ppl = _ppl(model, tokenizer, baseline_prefix, _REFERENCE)
    assert baseline_ppl < 50, baseline_ppl

    model.extend_context_length(
        64, rope_theta=10000.0, rope_scaling="linear", rope_scaling_factor=2.0
    )
    long_prefix = _filler_tokens(tokenizer, 48)
    assert len(tokenizer.encode(long_prefix)) == 48
    assert len(tokenizer.encode(long_prefix)) + len(tokenizer.encode(_REFERENCE)) <= 64
    extended_ppl = _ppl(model, tokenizer, long_prefix, _REFERENCE)

    assert extended_ppl < baseline_ppl * 1.05, (baseline_ppl, extended_ppl)


def test_invalid_rope_scaling_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=128, rope_scaling="yarn", rope_scaling_factor=2.0).validate()
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=128, rope_scaling="linear", rope_scaling_factor=1.0).validate()
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=128, rope_theta=0).validate()


def test_long_benchmark_files_exist() -> None:
    for name in ("needle.jsonl", "long_ppl.jsonl", "MANIFEST.json"):
        path = Path("data/benchmarks/bangla/long") / name
        assert path.exists(), name
    records = [
        json.loads(l)
        for l in (Path("data/benchmarks/bangla/long/needle.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) >= 50
    assert all(r["task"] == "needle" for r in records)
    assert max(r["context_len"] for r in records) == 8192

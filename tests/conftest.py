"""Shared pytest fixtures for the whole test suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

API_PORT = 8011
API_TEST_TOKEN = "test-token-abc123"


@pytest.fixture(scope="module")
def server():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.api.app:app", "--port", str(API_PORT)],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "KOTHAGPT_API_TOKEN": API_TEST_TOKEN},
    )
    try:
        for _ in range(50):
            try:
                httpx.get(f"http://localhost:{API_PORT}/health")
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        yield f"http://localhost:{API_PORT}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory):
    """A tiny trainable model/training/data config shared by trainer tests."""
    from ml.models.config import BaseModelConfig, DataConfig, ModelConfig, TrainingConfig
    from ml.tokenizer import train_bpe

    tmp = tmp_path_factory.mktemp("cfg")
    corpus = tmp / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"text": f"বাংলা ভাষা এটি নমুনা বাক্য সংখ্যা {i}।"}, ensure_ascii=False)
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    texts = [json.loads(line)["text"] for line in corpus.read_text(encoding="utf-8").splitlines()]
    tok_dir = tmp / "tok"
    tokenizer = train_bpe(texts, vocab_size=200, min_frequency=1)
    tokenizer.save(tok_dir)

    return BaseModelConfig(
        model=ModelConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
            max_position_embeddings=8,
        ),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-2,
            max_steps=4,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            mixed_precision="none",
        ),
        data=DataConfig(train=str(tok_dir), tokenizer_path=str(tok_dir / "tokenizer.json")),
    )


@pytest.fixture(scope="module")
def tokenizer(config):
    from ml.tokenizer import load_tokenizer

    return load_tokenizer(config.data.tokenizer_path)


def make_corpus(tmp_path: Path, docs: int = 8) -> Path:
    """Write a small JSONL corpus for trainer tests."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"text": f"বাংলা ভাষা এটি নমুনা বাক্য সংখ্যা {i}।"}, ensure_ascii=False)
            for i in range(docs)
        )
        + "\n",
        encoding="utf-8",
    )
    return path

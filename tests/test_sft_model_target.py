import json
import sys
from pathlib import Path

import pytest


def test_sft_cli_writes_checkpoint_loadable_by_model_target(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import yaml

    from evals.run import ModelTarget
    from ml.instruction.dataset import load_jsonl
    from ml.models import KothaGPT
    from ml.models.config import BaseModelConfig, DataConfig, ModelConfig, TrainingConfig
    from ml.sft.cli import main
    from ml.tokenizer import train_bpe

    train_path = Path("tests/fixtures/instruction.jsonl")
    records = load_jsonl(train_path)
    tokenizer = train_bpe(
        [f"{record.instruction} {record.output}" for record in records],
        vocab_size=256,
        min_frequency=1,
    )
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")
    training_config = BaseModelConfig(
        model=ModelConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        ),
        training=TrainingConfig(
            batch_size=1,
            max_steps=1,
            warmup_steps=0,
            mixed_precision="none",
        ),
        data=DataConfig(
            train=str(train_path),
            tokenizer_path=str(tokenizer_path),
        ),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(training_config.to_dict()), encoding="utf-8")
    base_dir = tmp_path / "base-run"
    base_checkpoint_dir = base_dir / "checkpoints"
    base_checkpoint_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text(json.dumps(training_config.to_dict()), encoding="utf-8")
    base_path = base_checkpoint_dir / "best.pt"
    torch.save({"model_state": KothaGPT(training_config.model).state_dict()}, base_path)
    base_checkpoint_bytes = base_path.read_bytes()
    out_dir = tmp_path / "sft-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ml.sft.cli",
            "--train",
            str(train_path),
            "--tokenizer",
            str(tokenizer_path),
            "--config",
            str(config_path),
            "--base",
            str(base_path),
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--out",
            str(out_dir),
        ],
    )

    main()

    assert json.loads((out_dir / "metrics.json").read_text())["steps"] == 1
    assert (out_dir / "checkpoints" / "best.pt").is_file()
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["base_checkpoint_sha256"]
    assert base_path.read_bytes() == base_checkpoint_bytes
    base_target = ModelTarget(base_path, tokenizer_path, max_new_tokens=2)
    assert isinstance(base_target.generate("<user>\nHi\n<assistant>\n", {}), str)
    target = ModelTarget(
        out_dir,
        tokenizer_path,
        max_new_tokens=2,
    )
    assert isinstance(target.generate("<user>\nHi\n<assistant>\n", {}), str)


def test_model_target_uses_checkpoint_engine(monkeypatch):
    from evals.run import ModelTarget
    from ml.inference import engine

    class FakeEngine:
        def __init__(self, **kwargs):
            self.options = kwargs

        def generate(self, prompt, *, max_new_tokens):
            assert prompt == "prompt"
            assert max_new_tokens == 7
            yield " answer "

    monkeypatch.setattr(engine, "KothaGPTEngine", FakeEngine)
    target = ModelTarget("base", "tokenizer", max_new_tokens=7)
    assert target.generate("prompt", {}) == "answer"
    assert target.engine.options["checkpoint"] == "base"

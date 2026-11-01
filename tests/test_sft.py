import pytest

from ml.sft.templates import apply_chat_template, check_tokenizer_coverage, parse_chat_template


def test_template_round_trip():
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "হ্যালো"},
        {"role": "assistant", "content": "হাই"},
    ]
    for name in ["default", "kothagpt-bn", "kothagpt-en"]:
        text = apply_chat_template(msgs, template=name)
        parse_chat_template(text, template=name)
        # parse returns messages without trailing assistant prompt; we applied with 3 msgs including assistant, so apply adds extra assistant prefix
        # For this test use prompt-only messages
        prompt_msgs = msgs[:2]
        text2 = apply_chat_template(prompt_msgs, template=name)
        parsed2 = parse_chat_template(text2, template=name)
        assert parsed2 == prompt_msgs


def test_tokenizer_coverage(tmp_path):
    # fake tokenizer with encode
    class FakeTok:
        def __init__(self) -> None:
            self.vocab = {"<user>": 1, "<assistant>": 2, "<system>": 3, "<tool>": 4, "<eos>": 5, "<pad>": 6}

        def encode(self, x):
            return [1]

    cov = check_tokenizer_coverage(FakeTok())
    assert all(cov.values())


def test_sft_trainer_smoke():
    pytest.importorskip("torch")
    from ml.instruction.dataset import load_jsonl
    from ml.models import KothaGPT, load_config
    from ml.sft.trainer import run_sft_trainer
    from ml.tokenizer import load_tokenizer

    config = load_config("ml/configs/sft.yaml")
    # Use tiny vocab from fixtures if best not exists
    import pathlib

    tok_path = pathlib.Path("ml/tokenizer/artifacts/best/tokenizer.json")
    if not tok_path.exists():
        pytest.skip("no tokenizer")
    tok = load_tokenizer(tok_path)
    records = load_jsonl("tests/fixtures/instruction.jsonl")
    model = KothaGPT(config.model)
    metrics = run_sft_trainer(model, records[:4], tok, max_steps=2, batch_size=2, max_length=128)
    assert metrics["steps"] == 2
    assert metrics["loss"] > 0


def test_sft_resume_restores_training_state(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.instruction.dataset import load_jsonl
    from ml.models import BaseModelConfig, DataConfig, KothaGPT, ModelConfig, TrainingConfig
    from ml.sft.trainer import run_sft_trainer
    from ml.tokenizer import train_bpe
    from ml.trainer.checkpoint import latest_checkpoint, load_checkpoint

    records_path = "tests/fixtures/instruction.jsonl"
    records = load_jsonl(records_path)
    tokenizer = train_bpe(
        [f"{record.instruction} {record.output}" for record in records],
        vocab_size=256,
        min_frequency=1,
    )
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")
    config = BaseModelConfig(
        model=ModelConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        ),
        training=TrainingConfig(
            batch_size=1,
            learning_rate=1e-3,
            max_steps=4,
            warmup_steps=0,
            min_lr=1e-4,
            save_interval=4,
            mixed_precision="none",
        ),
        data=DataConfig(train=records_path, tokenizer_path=str(tokenizer_path)),
    )
    run_dir = tmp_path / "resumable"

    torch.manual_seed(7)
    resumed_model = KothaGPT(config.model)
    first = run_sft_trainer(
        resumed_model,
        records,
        tokenizer,
        max_steps=2,
        batch_size=1,
        learning_rate=config.training.learning_rate,
        max_length=24,
        config=config,
        out_dir=run_dir,
    )
    assert first["step"] == 2
    checkpoint = load_checkpoint(latest_checkpoint(run_dir))
    assert checkpoint["step"] == 2
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["scheduler_state"]

    resumed = run_sft_trainer(
        resumed_model,
        records,
        tokenizer,
        max_steps=4,
        batch_size=1,
        learning_rate=config.training.learning_rate,
        max_length=24,
        config=config,
        out_dir=run_dir,
        resume_from=run_dir,
    )
    assert resumed["steps"] == 2
    assert resumed["step"] == 4
    assert load_checkpoint(latest_checkpoint(run_dir))["step"] == 4

    torch.manual_seed(7)
    fresh_model = KothaGPT(config.model)
    fresh = run_sft_trainer(
        fresh_model,
        records,
        tokenizer,
        max_steps=4,
        batch_size=1,
        learning_rate=config.training.learning_rate,
        max_length=24,
        config=config,
        out_dir=tmp_path / "fresh",
    )
    assert fresh["step"] == 4
    for resumed_parameter, fresh_parameter in zip(
        resumed_model.parameters(), fresh_model.parameters()
    ):
        assert torch.equal(resumed_parameter, fresh_parameter)

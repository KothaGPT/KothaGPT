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

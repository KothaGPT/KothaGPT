from pathlib import Path

import pytest

from evals.run import MockTarget, _record_prompt, run_suite

_ROOT = Path(__file__).parent.parent
_SUITES = (
    "sft-bn",
    "sft-en",
    "sft-multilingual",
    "sft-code",
    "sft-reasoning",
    "sft-conversation",
    "sft-fc",
    "sft-tool-use",
)


@pytest.mark.parametrize("family", _SUITES)
def test_sft_family_suite_scores_dev_records(family):
    results = run_suite(
        _ROOT / f"evals/suites/{family}.yaml",
        split="dev",
        target=MockTarget(),
    )
    metrics = results["summary"][family]["metrics"]
    assert results["summary"][family]["instances"] == 3
    assert metrics["exact_match"]["mean"] == 1.0
    assert metrics["normalized_match"]["mean"] == 1.0
    if family == "sft-fc":
        assert metrics["function_call_match"]["mean"] == 1.0
    if family == "sft-tool-use":
        assert metrics["tool_call_match"]["mean"] == 1.0


def test_sft_suite_reads_metadata_test_split():
    results = run_suite(
        _ROOT / "evals/suites/sft-bn.yaml",
        split="test",
        target=MockTarget(),
    )
    assert results["summary"]["sft-bn"]["instances"] == 1


def test_sft_conversation_prompt_keeps_prior_assistant_turns():
    prompt = _record_prompt(
        {
            "messages": [
                {"role": "user", "content": "Help me plan dinner."},
                {"role": "assistant", "content": "What ingredients do you have?"},
                {"role": "user", "content": "Rice and eggs."},
            ],
            "output": "Make egg fried rice.",
        }
    )
    assert "What ingredients do you have?" in prompt
    assert prompt.endswith("<assistant>\n")


@pytest.mark.parametrize(
    ("family", "metric"),
    [("sft-fc", "function_call_match"), ("sft-tool-use", "tool_call_match")],
)
def test_sft_structured_scorers_reject_wrong_arguments(family, metric):
    class WrongCallTarget:
        def generate(self, prompt, record):
            return '{"name":"wrong","arguments":{}}'

    results = run_suite(
        _ROOT / f"evals/suites/{family}.yaml",
        split="dev",
        target=WrongCallTarget(),
    )
    assert results["summary"][family]["metrics"][metric]["mean"] == 0.0
import json
from pathlib import Path

from data.benchmarks.sft.generate import generate_all

_BENCHMARK_DIR = Path(__file__).parent.parent / "data/benchmarks/sft/v1"
_FAMILIES = {
    "sft-bn.jsonl",
    "sft-en.jsonl",
    "sft-multilingual.jsonl",
    "sft-code.jsonl",
    "sft-reasoning.jsonl",
    "sft-conversation.jsonl",
    "sft-fc.jsonl",
    "sft-tool-use.jsonl",
}


def test_sft_benchmark_families_are_valid_and_deterministic(tmp_path):
    manifest = generate_all(tmp_path / "generated")
    assert set(manifest["families"]) == _FAMILIES

    for filename, details in manifest["families"].items():
        generated = (tmp_path / "generated" / filename).read_bytes()
        assert generated == (_BENCHMARK_DIR / filename).read_bytes()
        records = [json.loads(line) for line in generated.decode("utf-8").splitlines()]
        assert len(records) == details["records"] == 4
        assert details["split_counts"] == {"dev": 3, "test": 1}
        ids = [record["metadata"]["record_id"] for record in records]
        assert len(set(ids)) == len(ids)
        assert all(
            record["metadata"]["family"] == filename.removesuffix(".jsonl") for record in records
        )

    regenerated = generate_all(tmp_path / "regenerated")
    assert regenerated == manifest

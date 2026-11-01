# SFT Benchmarks

`v1/` contains compact per-family evaluation sets in the same JSONL record
format accepted by `ml.instruction.dataset.load_jsonl`:

- `sft-bn.jsonl`: Bangla instructions
- `sft-en.jsonl`: English instructions
- `sft-multilingual.jsonl`: Bangla-English instructions
- `sft-code.jsonl`: coding
- `sft-reasoning.jsonl`: short verifiable reasoning
- `sft-conversation.jsonl`: multi-turn dialogue
- `sft-fc.jsonl`: function calls
- `sft-tool-use.jsonl`: tool selection and arguments

Each record includes `metadata.record_id`, `metadata.family`, and `metadata.split`.
The split is deterministic per family, with 3 dev and 1 test record in this
seed version. `manifest.json` records family counts, split counts, and file
digests. These small fixtures validate the benchmark pipeline; they are not a
release-quality statistical evaluation set.

Regenerate the files with:

```bash
make sft-benchmark-build
```
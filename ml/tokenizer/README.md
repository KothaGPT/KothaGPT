# Tokenizer — Phase 1B

Train and benchmark our own Bangla tokenizer (BPE, Unigram, and WordPiece), then
freeze the winner.

```
Bangla Corpus
    ↓
Normalizer (Phase 1A)
    ↓
Tokenizer Trainer (BPE / Unigram / WordPiece)
    ↓
Vocabulary
    ↓
Tokenizer
    ↓
Token Efficiency Benchmark
```

All three algorithms are implemented from scratch in pure stdlib Python (no
`sentencepiece` / `tokenizers` dependency) and are word-based with a GPT-2-style
leading `▁` space marker.

## Usage

```bash
# Train a single tokenizer
python -m ml.tokenizer.cli train \
  --corpus data/processed/$(cat data/processed/CURRENT)/train \
  --algorithm bpe --vocab-size 16000 --out ml/tokenizer/artifacts/bpe-16000

# Run the full experiment matrix: {bpe, unigram, wordpiece} x {16k, 32k, 50k}
python -m ml.tokenizer.cli experiments \
  --corpus data/processed/$(cat data/processed/CURRENT)/train \
  --algorithms bpe,unigram,wordpiece \
  --out ml/tokenizer/artifacts

# Encode / benchmark a saved tokenizer
python -m ml.tokenizer.cli encode --tokenizer ml/tokenizer/artifacts/best/tokenizer.json --text "বাংলা ভাষা" --show-tokens
python -m ml.tokenizer.cli benchmark --tokenizer ml/tokenizer/artifacts/best/tokenizer.json
```

The corpus argument accepts a Phase 1A shard directory, a single `.txt` file, or
a `.jsonl`/`.jsonl.gz` file (see `ml/tokenizer/corpus.py`).

## Experiments

`experiments` trains every `{algorithm} x {vocab_size}` combination, benchmarks
each on five test sets, and writes:

- `ml/tokenizer/artifacts/experiments/<algo>-<vocab>/tokenizer.json`
- `ml/tokenizer/artifacts/comparison.json` — machine-readable results
- `ml/tokenizer/artifacts/REPORT.md` — the comparison table
- `ml/tokenizer/artifacts/best/` — the frozen best tokenizer

The benchmark covers the Phase 1B test matrix:

- `bangla` — plain Bangla prose
- `mixed` — Bangla/English mixed text
- `punctuation` — punctuation-heavy text
- `emoji` — Bangla with emoji
- `code` — code with Bangla comments

Metric: **tokens per character** (lower = better). A frozen `best` tokenizer is
selected by lowest average across the five sets.

## Details

- **BPE** (`bpe.py`) — word-frequency-counted merges (subword-nmt style) with a
  lazy heap; decoding re-merges by lowest merge rank (GPT-2 style).
- **Unigram** (`unigram.py`) — substring candidate vocab, EM re-estimation with
  Viterbi best-path counts, pruning to the target vocab size; encoding via
  Viterbi shortest path over a trie.
- **WordPiece** (`wordpiece.py`) — merges selected by the likelihood increase
  `freq(ab) / (freq(a) * freq(b))`; encoding via greedy longest-match-first over
  the merged subwords.
- All three seed the vocabulary with the full Bengali block, the Bangla danda `।`
  (U+0964), ASCII, and common punctuation so unseen characters still tokenize
  instead of collapsing to `<unk>`. Emoji are intentionally not in the vocab.

Artifacts are git-ignored; never commit trained vocabularies or weights.

## Tests

```bash
python -m pytest tests/test_tokenizer.py
```
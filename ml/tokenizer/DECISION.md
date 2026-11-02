# Decision — Tokenizer & Vocabulary Freeze

- **date**: 2026-08-20T03:43:44+00:00
- **algorithm**: `bpe` (vocab 16,000) — canonical production vocab
- **corpus digest**: `7bf1a6e740e3`

## Why this tokenizer/vocab

- Trained on the normalized Bangla corpus (33,319 docs of 133,275, stride 4;
  coverage measured on a 5,000-doc sample).
- Coverage: **100.00%** of script characters on a 5,000-doc sample; unk rate 0.00%.
- Efficiency gate passes: tpc 0.3361 (<= 2.0), unk 0.00% (<= 0.005), fidelity 100% (>= 1.0).

## Vocab-size comparison (same corpus, same stride-4 sample)

| vocab size | version                                  | tokens/char | script-char coverage | unk  | decode fidelity | gate |
| ---------- | ---------------------------------------- | ----------- | -------------------- | ---- | --------------- | ---- |
| 16,000     | `1.0.0+7bf1a6e740e3.5d53c485`            | 0.3361      | 100.00%              | 0.0% | 100%            | pass |
| 32,000     | `1.0.0+7bf1a6e740e3.c5df90bc`            | 0.2790      | 100.00%              | 0.0% | 100%            | pass |
| 50,000     | `1.0.0+7bf1a6e740e3.29927a15`            | 0.2720      | 100.00%              | 0.0% | 100%            | pass |

Note: 32k/50k coverage re-measured with the whitespace-aware metric; the
earlier 98.43/98.48% figures were inflated gaps caused by counting newlines.
Larger vocabs shave tokens/char (~19% from 16k→32k, then marginal) at the cost
of a heavier embedding table and slower sampling. The 16k vocab is kept as the
canonical production vocab (committed under `ml/tokenizer/vocab/`); 32k/50k
runs are reference points.

## Reference baseline (dev-only)

`sentencepiece` 16k BPE, trained on a 5,000-doc subsample of the same corpus
(its own normalization), measured on the same gated dev sets:
`tokens/char` **0.2814** vs pure-stdlib 16k **0.2991** (~6% gap). The pure-stdlib
tokenizer stays canonical: no compiled dependency, word-marker + translit
pipelines are preserved, and all gates (unk, decode fidelity, coverage) pass.

## Open questions

- Evaluate sentencepiece at 32k/50k for a full-reference comparison (optional).

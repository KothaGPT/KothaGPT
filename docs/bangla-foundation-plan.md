# Bangla Language Foundation — Implementation Plan

Goal: turn the Bangla data pipeline and Phase 1B tokenizer experiments into a
frozen, production-grade language foundation: a canonical Bangla normalizer, a
selected + frozen tokenizer and vocabulary, transliteration support, and the
first reproducible Bangla benchmark dataset with an evaluation harness.

Guiding principles:

- Build on what exists (`data/pipeline`, `ml/tokenizer`, `evals/`) — no rewrite.
- Every step ships a measurable artifact (normalizer module, frozen vocab,
  comparison report, benchmark dataset, eval report).
- The tokenizer/vocab freeze happens **after** normalization + transliteration
  land, so the vocabulary reflects the final input pipeline.
- Success metrics are objective and tracked in `docs/` + `TODO`.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Tokenizer training | `ml/tokenizer/{bpe,unigram,base,corpus,cli,benchmark}.py` (pure stdlib, GPT-2-style `▁` marker) | no frozen winner, no reference comparison (sentencepiece), no char/word baselines |
| Unicode normalization | `data/pipeline/normalize.py::unicode_normalize` (NFC, ZWJ-preserving) | no NFKC policy for Latin, no Y-combining / vowel-sign edge cases, not exposed as a reusable package |
| Punctuation normalization | only NBSP/control cleanup | no dedicated Bangla punctuation module (danda variants, quotes, digits, dashes) |
| Bangla-English mixed text | `benchmark.py` "mixed" test set only | no normalization/segmentation rules for code-switching |
| Transliteration | none | entire workstream missing |
| Vocabulary | trained ad hoc per experiment | no published/frozen `vocab.json`, no coverage report |
| Token efficiency tests | `benchmark.py` (5 sets, tokens/char) | needs more sets + unknown-rate + decode-fidelity + baselines |
| Benchmark dataset | `evals/suites/bangla.yaml` (task names only) | no data files, no instances |
| Language evaluation | `evals/README.md` stub | no runner/harness, no metrics implementation |

---

## Workstreams

### WS-1 — Bangla tokenizer: create / select  (`ml/tokenizer`)

Goal: pick and freeze the tokenizer.

- Finish the experiment matrix `{bpe, unigram} × {16k, 32k, 50k}` on a validated
  corpus (existing `experiments` command).
- Add reference baselines for context: char-level, word-level, and an optional
  `sentencepiece`/`tokenizers` run (dev-only dependency, kept out of the
  training path) so the pure-stdlib result is comparable.
- Add selection criteria beyond tokens/char:
  - unknown-token rate on the dev set (target `< 0.5%`),
  - decode round-trip fidelity (encode→decode == original, target 100% on dev),
  - compression ratio vs a byte-level baseline.
- Deliverables: `ml/tokenizer/artifacts/best/` (frozen), `REPORT.md` with the
  decision rationale, `ml/tokenizer/DECISION.md`.
- Metric: lowest tokens/char **and** the thresholds above; selection documented.

### WS-2 — Bangla Unicode normalization  (`data/pipeline` -> shared package)

Goal: canonical Unicode normalization for Bangla.

- Extend `unicode_normalize` with:
  - NFC canonical for Bangla; explicit NFKC path for Latin-heavy segments.
  - Y-combining fixes: normalize the historical `{ka}+ZWJ+{ya}`-style sequences
    and the `ে+্+র` / `েয` reordering cases used in Bengali orthography.
  - Bangla digit policy (keep Bengali digits, or map to ASCII — configurable).
- Promote normalization to a reusable library used by both the dataset pipeline
  and the tokenizer (add `packages/core` entry or `ml/tokenizer/normalizer.py`
  re-export).
- Deliverables: extended `normalize.py` + unit tests (`tests/test_normalize.py`),
  documented policy (`docs/bangla-normalization.md`).
- Metric: round-trip stable; corpus unaffected-codepoint rate `> 99.9%`.

### WS-3 — Bangla punctuation normalization  (new `data/pipeline/punctuation.py`)

Goal: canonical punctuation mapping.

- Normalize:
  - danda variants (`।` U+0964, `॥` U+0965) -> single canonical danda + spacing rule.
  - Bangla/English quotes (`’`, `‘`, `“`, `”`, `"`, `'`) -> canonical pair.
  - dashes (`—`, `–`, `-`) policy; ellipsis (`…` vs `...`).
  - digits (Bengali ০-৯ vs Arabic-Indic ٠-٩ vs ASCII 0-9) -> one canonical set.
  - zero-width joiner policy already in `normalize.py` (keep for conjuncts);
    remove zero-width non-joiner unless meaningful.
- Deliverables: `data/pipeline/punctuation.py` + tests, applied inside
  `normalize_text` by default and configurable.
- Metric: 100% of a curated punctuation fixture maps to canonical forms.

### WS-4 — Bangla–English mixed text handling  (new `data/pipeline/mixed.py`)

Goal: rules for code-switched Bangla/English text.

- Handling rules:
  - word-boundary segmentation across scripts (Bangla + Latin) without splitting
    romanized Bengali words; keep Latin words intact for the tokenizer.
  - spacing policy around script switches (do not merge a Latin word and a Bangla
    word); the existing `▁` marker handles it at token level.
  - camelCase / acronym handling for Latin words embedded in Bangla sentences.
  - a detector (`is_bangla`, `script_ratio`) exposed for downstream QA/RAG.
- Deliverables: `data/pipeline/mixed.py` + tests; extend the `benchmark.py`
  "mixed" set with realistic code-switched Bangla (social-media style).
- Metric: mixed-set tokenization shows no `unk` increase vs plain Bangla; a
  golden mixed corpus is byte-identical after normalization round-trip.

### WS-5 — Bangla transliteration handling  (new `ml/tokenizer/transliterate.py`)

Goal: lossless Latin<->Bangla transliteration for romanized Bangla input.

- Implement a deterministic rule-based transliterator (Avro-style / ISO 15919
  subset) covering Bengali consonants/vowels, conjuncts via ZWJ, `ং ঃ ঁ`, and
  common romanization variants (e.g., `sh/sh/kh`, `o/o`).
- Scope: normalize romanized input to Bangla before tokenization; provide
  reverse (Bangla->Latin) for evaluation display; add a `--transliterate` flag
  to `encode` and the CLI `chat`.
- Deliverables: `ml/tokenizer/transliterate.py` + `tests/test_transliterate.py`,
  a variant table (`docs/bangla-transliteration.md`), benchmark set `translit`.
- Metric: >= 95% of a curated 500-pair romanization fixture maps to the expected
  Bangla form; `unk` on transliterated input drops to ~plain-Bangla level.

### WS-6 — Bangla vocabulary creation  (new `ml/tokenizer/vocab.py` + freeze)

Goal: publish the canonical vocabulary.

- After WS-2..WS-5, retrain the winner tokenizer on the normalized corpus.
- Produce `ml/tokenizer/vocab/vocab.json` (token->id), `vocab.md` (size, coverage,
  top-N freq), and a coverage report on dev (`unk`, OOV by script).
- Version the vocab (semantic version or dataset-hash suffix) so SDK/API model
  cards reference `vocab=<version>`.
- Deliverables: `vocab/vocab.json` + `vocab/REPORT.md`, `ml/tokenizer/vocab.py`
  helper (export/validate/version).
- Metric: coverage `> 99.5%` on dev; vocabulary size = target (16k/32k/50k) with
  an OOV budget documented.

### WS-7 — Token efficiency testing  (extend `ml/tokenizer/benchmark.py`)

Goal: a defensible efficiency benchmark.

- Extend test sets: `translit` (from WS-5), `digits`, `names` (Bangla proper
  nouns + English loanwords), `social` (emoji-heavy, code-switched).
- Add metrics: unknown rate, decode round-trip fidelity, compression vs byte and
  char baselines, paragraph-level stability.
- Wire into CI (`make tokenizer-check` + a new `make tokenizer-bench` gate) so
  regressions fail the build.
- Deliverables: extended `benchmark.py`, `ml/tokenizer/artifacts/benchmark.json`,
  CI gate.
- Metric: frozen tokenizer meets all thresholds; CI gate blocks regressions.

### WS-8 — Bangla benchmark dataset creation  (new `data/benchmarks/bangla/`)

Goal: first curated Bangla benchmark dataset.

- Create seed tasks that map to `evals/suites/bangla.yaml`:
  - `bangla_qa`: extractive/MCQ QA from validated Bangla prose (hand-authored
    seed + sourced reading passages), JSONL `{id, context, question, answers}`.
  - `bangla_generation`: prompts + reference continuations.
  - `bangla_translation`: en->bn / bn->en sentence pairs (curated, license-checked).
  - `bangla_summarization`: article–summary pairs.
- Version + split (dev/test), store under `data/benchmarks/bangla/v1/`.
- Deliverables: JSONL datasets + `README.md` (licenses, statistics) + a generator
  script; `evals/suites/bangla.yaml` extended with instance counts.
- Metric: >= 500 QA instances, >= 1k translation pairs, >= 100 summarization
  pairs; PII/license checks pass (reuse `data/pipeline` filters).

### WS-9 — Bangla language evaluation  (new `evals/` runner)

Goal: reproducible evaluation for the benchmark.

- Implement an eval runner (`evals/run.py` + `evals/metrics.py`) that:
  - loads `evals/suites/bangla.yaml` + `data/benchmarks/bangla/`,
  - runs tasks against a target (mock backend now, real model later),
  - computes `exact_match`, `rouge`, and `language_detection` /
    `bengali_script_ratio` sanity metrics.
- Emit `evals/results/<date>-bangla.json` + `REPORT.md`.
- Deliverables: runner + metrics + sample results on the mock backend; CI hook
  (`make eval-bangla`).
- Metric: eval completes on seed data; metrics reported with confidence; runs
  reproducible (fixed seeds/dates).

---

## Sequencing (dependent ordering)

    WS-2 Unicode -> WS-3 Punctuation -> WS-4 Mixed -> WS-5 Transliteration
         |                                |               |
         +--------------> WS-1 retrain + select <----------+
                                  |
                                WS-6 Vocab freeze
                                  |
                                WS-7 Efficiency gate
                                  |
                    WS-8 Benchmark dataset -> WS-9 Eval harness

- **Wave A (language preprocessing):** WS-2 -> WS-3 -> WS-4 -> WS-5 (each with tests).
- **Wave B (tokenizer freeze):** WS-1 (retrain) -> WS-6 -> WS-7.
- **Wave C (evaluation):** WS-8 -> WS-9.

## Success metrics (project-wide)

- Tokenizer: avg tokens/char <= current best; unk < 0.5%; decode fidelity 100%.
- Normalization: canonical policy documented; corpus unaffected-codepoint > 99.9%.
- Transliteration: >= 95% fixture accuracy; romanized input unk ~ Bangla-level.
- Vocabulary: coverage > 99.5% on dev; versioned artifact.
- Benchmark: >= 500 QA / >= 1k translation / >= 100 summarization instances, PII-safe.
- Evaluation: reproducible reports; CI gates (`make tokenizer-bench`, `make eval-bangla`).

## Owners & risks

- Owners: `ml/tokenizer` (WS-1, WS-5, WS-6, WS-7), `data/` (WS-2, WS-3, WS-4, WS-8), `evals/` (WS-9).
- Risks & mitigations:
  - Transliteration ambiguity -> curate a golden variant table and lock it.
  - NFKC breaking Bangla conjuncts -> keep NFC default for Bangla; NFKC only for
    Latin segments; golden round-trip test.
  - Benchmark license/PII -> reuse pipeline license + PII filters and record sources.
  - Reference tokenizer (sentencepiece) dependency -> optional dev-only; never
    required in the training path.

## Status & next steps

Wave A (WS-2 Unicode normalization, WS-3 punctuation, WS-4 mixed text, WS-5
transliteration) is implemented and tested (`tests/test_bangla_foundation.py`,
`tests/fixtures/transliteration.json`). Wave B and Wave C are also implemented:

1. **WS-7** — `benchmark.py` has the `translit`/`digits`/`names`/`social` sets
   and unk-rate / decode-fidelity / compression / paragraph-stability metrics;
   the CI gate `make tokenizer-bench` (unk < 0.5%, fidelity 100%, tokens/char
   budget) runs in `.github/workflows/ci.yml`.
2. **WS-8** — `data/benchmarks/bangla/v1/` is generated deterministically by
   `generate.py` (576 QA with verbatim-span answers, 1120 bn↔en translation,
   108 summarization, 26 generation) with MANIFEST + README; counts live in
   `evals/suites/bangla.yaml`.
3. **WS-9** — `evals/run.py` + `evals/metrics.py` (exact_match, rouge,
   language_detection, bengali_script_ratio, mean + 95% CI) run the suite
   against a mock backend (gold = 100%) or the API backend; results go under
   `evals/results/`; `make eval-bangla` runs in CI.
4. **WS-1/WS-6** — `ml/tokenizer/vocab.py` (export/validate/version) and the
   `freeze` CLI command produce `ml/tokenizer/artifacts/best/`,
   `ml/tokenizer/vocab/` (`vocab.json` + coverage report, versioned
   `1.0.0+<corpus-hash>`), `ml/tokenizer/artifacts/REPORT.md` and
   `ml/tokenizer/DECISION.md`. The CI freeze runs on the 12-doc fixture corpus;
   `--sample-stride N` keeps training memory-safe on the real corpus.

Remaining:

- **Done (2026-08-20):** production 16k BPE freeze on the normalized Bangla
  Wikipedia corpus (133,275 docs; stride-4 sample) → `ml/tokenizer/vocab/`
  version `1.0.0+7bf1a6e740e3.5d53c485`, **100% script-char coverage**, 0% unk,
  tpc 0.3361, decode fidelity 100% (gate passed). 32k/50k re-freezes done as
  reference points and recorded in `ml/tokenizer/DECISION.md` (tpc 0.279 /
  0.272, all gates pass); 16k remains the canonical production vocab.
  sentencepiece 16k reference baseline measured (dev tpc 0.2814 vs 0.2991) and
  recorded in `ml/tokenizer/DECISION.md`. Coverage metric fixed to ignore
  whitespace (newlines were artificially depressing it to ~98.5%).
- Re-check 32k/50k sentencepiece reference points if desired (optional).
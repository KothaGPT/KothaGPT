# Instruction Tuning (SFT) — Implementation Plan

Goal: take the pre-trained KothaGPT base model (`ml/pretrain/artifacts/`) and the
processed corpus tooling and produce instruction-tuned variants — Bangla-first,
English, multilingual, coding, reasoning, conversation, and function-calling /
tool-use — with an SFT framework, a versioned instruction dataset stack, and a
release-grade evaluation gate.

Guiding principles:

- Build on what exists: `ml/trainer/` already implements the loop, mixed
  precision, DDP/FSDP, checkpoints, scheduler, and monitoring; `ml/models/` is the
  frozen base architecture. SFT layers *task adaptation* on top: instruction
  format handling, chat templating, dataset mixing, and eval.
- Every SFT run is reproducible and resumable: pinned to (base checkpoint,
  data version, tokenizer digest, chat template) and recorded in
  `ml/sft/artifacts/<run-id>/`.
- Data provenance matters as much as weights: every instruction record carries
  `{source, license, quality_score}` and passes the existing Phase-B gates
  (`data/pipeline/`).
- Correctness first: toy CPU smoke runs gate every change; eval gates the
  release, not the eyeball.
- PyTorch stays a training-only dependency; never import `torch` from the API
  service.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Base model | `ml/models/` full KothaGPT (embedding, attention, blocks, SwiGLU, RoPE, RMSNorm, output head) | no pretrained checkpoint is present in this checkout (pretraining WS-12 pending) |
| Trainer core | `ml/trainer/` loop, checkpoint (atomic/resume), scheduler, monitor, DDP/FSDP, mixed precision | SFT still uses a simpler single-process loop rather than the resumable core |
| SFT package | `ml/sft/` templates, assistant-masked dataset, trainer, and CLI; CLI writes inference-loadable checkpoints and provenance metadata | SFT loop does not yet reuse the full resumable trainer/checkpoint path |
| Preference tuning | `ml/preference/` DPO dataset and trainer | out of scope (follow-up plan) |
| Data pipeline | `data/pipeline/` prose and instruction paths with normalization, provenance/license, safety, deduplication, scoring, splitting, and versioned shards | no production instruction corpus is checked in; supply curated/reviewed source JSONL |
| Eval | SFT family suites, checkpoint-backed `model` target, and base-vs-SFT non-regression command (`make sft-gate`) | checked-in SFT v1 data is a tiny pipeline fixture, not a release benchmark; no real checkpoint/data is present |
| Tokenizer | frozen 16k BPE (`ml/tokenizer/artifacts/best/`, GPT-2-style `▁`) | no chat special tokens (e.g. `<&#124;im_start&#124;>`) verified in vocab |
| Configs | `ml/configs/sft*.yaml` and `ml/configs/small.yaml` | production data/model artifacts must be supplied by the run owner |

---

## Workstreams

### WS-1 — Instruction dataset তৈরি (`data/synthetic/` + `data/pipeline/schema.py`)

Goal: a versioned, quality-gated, Bangla-first instruction corpus that feeds
every downstream WS.

- Curate JSONL records as either `{instruction, input?, output, source,
  license}` or `{messages: [{role, content}], source, license}`. The
  `kothagpt-data instruction run` pipeline canonicalizes both forms and applies
  provenance, license, PII, spam, toxic-content, exact/optional near-dedup,
  and deterministic split gates before emitting shards. Synthetic records must
  be reviewed and use the same schema/gates; no model is invoked by the
  pipeline.
- Add `data/pipeline/schema.py` validating
  `{messages: [{role: system|user|assistant, content}], source, license,
  quality_score}`; reject malformed or empty-answer records.
- Run through the Phase-B gates and the deterministic, model-free quality
  scorer (`--min-score`) before a release; land shards at
  `data/processed/<ver>/instruction/{train,validation,test}/`.
- Emit per-task-type tags (`general`, `reasoning`, `coding`, `conversation`,
  `function_call`, `tool_use`) so WS-3..WS-10 can mix selectively.
- Deliverables: `data/synthetic/` generators, `schema.py`, instruction shards +
  `MANIFEST.json` with quality histogram.
- Metric: instruction shard passes all gates with zero PII/license failures;
  all answers non-empty; ≥ threshold quality score; score histogram and
  percentiles are recorded; reproducible from config.

### WS-2 — SFT pipeline তৈরি (`ml/sft/`)

Goal: a chat-template-aware, maskable SFT trainer wrapping `ml/trainer/`.

- New package `ml/sft/`:
  - `templates.py` — chat template registry (Bangla/English, system+turn
    formatting, `apply_chat_template` / `parse_chat_template` round-trip), plus
    special-token checks against the frozen tokenizer.
  - `dataset.py` — instruction dataset loader (mixing by task type + source,
    weights, oversampling); yields `(input_ids, labels)` with **assistant-mask
    loss** (only assistant tokens contribute to CE).
  - `train.py` — CLI `python -m ml.sft.train run --config ml/configs/sft.yaml
    --base ml/pretrain/artifacts/<run-id>/checkpoints/best.pt`, reusing
    `ml/trainer/` loop/checkpoint/scheduler/monitor; EOS-padded packing.
  - `mix.py` — cross-task mixture builder + contamination check (no eval-set
    overlap with any training shard).
- Add `ml/configs/sft.yaml` (base checkpoint path, template, mix, LR with SFT
  scale ~1e-5..1e-4, epochs, max length, eval split).
- Deliverables: `ml/sft/*`, `ml/configs/sft.yaml`, `make sft-smoke`.
- Metric: a CPU smoke run (`--max-steps 20`) on a toy instruction shard
  completes; assistant-mask loss starts from the base-model loss and decreases;
  template round-trip is byte-identical; tokenizer covers all chat special
  tokens.

### WS-3 — বাংলা instruction tuning

Goal: the flagship deliverable — a Bangla-instruct variant.

- Curate the Bangla-heavy mix (≥ 70% Bangla) from WS-1 tagged records +
  verified Bangla QA/benchmark-derived examples (`data/benchmarks/bangla/v1`).
- Train from `best.pt` with SFT config; log per-task-type val loss.
- Deliverables: `ml/sft/artifacts/<run-id>/` (checkpoints, history, config),
  report.
- Metric: Bangla instruction-following eval (WS-11) ≥ documented threshold and
  non-worse than base on Bangla v1 benchmarks; sample generations read natural
  Bangla.

### WS-4 — English instruction tuning

Goal: parity English-instruct capability on the same base model.

- Build a curated English mix (translated/curated Alpaca-class data, cleaned)
  plus English portion of WS-1.
- Deliverables: English-instruct config `ml/configs/sft-en.yaml` + run artifacts.
- Metric: English IF eval ≥ threshold; no regression on Bangla evals after
  multi-task mix (coordinate with WS-5).

### WS-5 — Multilingual instruction tuning

Goal: a single model that serves bn + en (+ optional `hi` later) without
forgetting either.

- Combine WS-3 + WS-4 mixtures with **loss-weighted balance** (e.g. Bangla
  upweighted) and a per-language eval schedule during training.
- Extend `evals/suites/` with per-language IF sets; log losses per language
  group.
- Deliverables: `ml/configs/sft-multilingual.yaml`, per-language eval report.
- Metric: multilingual model within X% of each single-language model on its
  home-language eval while staying above base-model baselines.

### WS-6 — Coding instruction tuning

Goal: Bangla-adjacent + general coding instruction capability.

- Add a code instruction shard (The Stack `bn`-relevant subset or curated
  GitHub issues/commits → instruction form; per `docs/dataset-pipeline-plan.md`
  A6, code keeps its own shard stream and dedup).
- Separate mixture so prose and code don't degrade each other; include
  Bangla-comment coding examples.
- Deliverables: code mix config + `ml/configs/sft-code.yaml`.
- Metric: code IF eval (function synthesis, bug fix, explanation) ≥ threshold;
  code shard passes dedicated code dedup + quality gates.

### WS-7 — Reasoning dataset তৈরি

Goal: verifiable step-by-step reasoning data, not just longer answers.

- Generate/curate reasoning pairs (math word problems, logic, commonsense,
  Bangla + English) with **verifiable intermediate steps**; store rationale +
  final answer, and keep chain-of-thought optional at inference (no hidden
  scratchpad leakage in eval).
- Add a synthetic verifier (answer-extraction + self-consistency check) to
  reject unsound rationales.
- Deliverables: reasoning shard (`task_type=reasoning`) + verifier script.
- Metric: ≥ threshold verifier agreement on held-out reasoning samples;
  records carry checkable final answers.

### WS-8 — Conversation tuning

Goal: coherent multi-turn dialogue beyond single-shot instruction.

- Extract/template multi-turn conversations (≥ 2 turns, license-checked) and
  synthetic dialogue chains (dataset-pipeline A9); schema validated by
  `data/pipeline/schema.py`.
- Masked-loss over assistant turns only; train on a conversation-weighted mix.
- Deliverables: `ml/configs/sft-conversation.yaml` + conversation shard.
- Metric: 2+ turn dialogue eval (coherence, on-topic, turn-role correctness);
  masked loss excludes user/system tokens.

### WS-9 — Function-calling dataset

Goal: teach the model to emit structured function calls.

- Design a small Bangla/English function-call schema
  (`<function_call>` JSON or typed tokens) and a deterministic
  template→verification generator (given a function spec + utterance → correct
  call), plus edge cases (missing args, ambiguous intent → ask clarification).
- Verifier: parse the emitted call and check against the gold call (args exact
  match).
- Deliverables: function-call shard + `verify_function_call.py`.
- Metric: exact-arg match rate ≥ threshold on a held-out set; zero calls on
  non-call intents (no false triggers).

### WS-10 — Tool-use dataset

Goal: teach the model to call tools and integrate results into the answer.

- Build a tool-recall dataset: user asks with a needed tool; model emits the
  call, receives a mocked tool result, then produces the final answer (2-round
  template).
- Deterministic tool registry (calculator, weather/search mocks, DB lookup) so
  evals are reproducible offline; verifier checks call correctness + answer
  grounded in the returned result.
- Deliverables: tool-use shard + mock tool registry + verifier.
- Metric: call correctness ≥ threshold AND answer grounded in tool result
  (contains expected fact) ≥ threshold.

### WS-11 — SFT evaluation

Goal: a release gate that measures what tuning bought, per task family.

Implemented: `evals/run.py --target model` loads local KothaGPT checkpoints and
uses training-compatible chat prompts. `make sft-real` requires an explicit
training JSONL and pretrained checkpoint, writes the tuned artifact, then runs
`make sft-gate` against the base on each SFT family. The gate rejects
regressions in exact/normalized matching and function/tool-call equality.
`make eval-sft-ci` remains a fixture-based harness smoke test; it is not the
real-model release gate.

The checked-in SFT benchmark has only four records per family (three dev, one
test), so gate results are wiring checks and must not be used to claim
statistical release quality. A production release requires a larger,
independently curated held-out benchmark and actual pretrained/training
artifacts, neither of which is checked into this repository.

Example real run (use a checkpoint from the pretraining artifact directory and
an independently curated instruction JSONL):

```bash
make sft-real variant=bn \
  train_data=/path/to/instruction-train.jsonl \
  base=ml/pretrain/artifacts/<run-id>/checkpoints/best.pt \
  eval_data=/path/to/held-out-sft-benchmarks \
  device=cuda
```

`eval_data` must contain the family JSONL files used by the SFT suites and is
required by `make sft-real`. A direct `make sft-gate` without `eval_data` uses
the checked-in fixture benchmark and prints its small-sample warning.

---

## Sequencing & dependencies

```text
WS-1 instruction dataset ──> WS-2 SFT pipeline (CPU smoke gates GPU work)
                                  │
WS-7 reasoning data ──────────────┤
WS-8 conversation data ───────────┤  (data WS-7..10 feed WS-2's mix)
WS-9 function-calling data ───────┤
WS-10 tool-use data ──────────────┤
                                  ├──> WS-3 Bangla ──┐
                                  ├──> WS-4 English ──┼─> WS-5 multilingual
                                  ├──> WS-6 coding ───┘
WS-11 SFT evaluation ─────────────┘ (gates WS-3..WS-10 releases)
```

WS-1 → WS-2 is the critical path. WS-3 (Bangla, flagship) runs first after
WS-2; WS-4/WS-6 are parallel; WS-5 depends on WS-3+WS-4. WS-7..WS-10 build
task-specific data in parallel and are tuned after their shards land. WS-11 is
a parallel, always-running gate and the summit.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Instruction dataset তৈরি | WS-1 |
| SFT pipeline তৈরি | WS-2 |
| বাংলা instruction tuning | WS-3 |
| English instruction tuning | WS-4 |
| Multilingual instruction tuning | WS-5 |
| Coding instruction tuning | WS-6 |
| Reasoning dataset তৈরি | WS-7 |
| Conversation tuning | WS-8 |
| Function-calling dataset | WS-9 |
| Tool-use dataset | WS-10 |
| SFT evaluation | WS-11 |

## Tests

```bash
make data && python -m pytest tests/test_instruction_schema.py   # WS-1
make sft-smoke && python -m pytest tests/test_sft.py              # WS-2
make sft-real variant=bn train_data=/path/to/train.jsonl base=/path/to/base.pt  # WS-3/11
make sft-real variant=en train_data=/path/to/train.jsonl base=/path/to/base.pt  # WS-4
make sft-real variant=multilingual train_data=/path/to/train.jsonl base=/path/to/base.pt  # WS-5
make sft-real variant=code train_data=/path/to/train.jsonl base=/path/to/base.pt  # WS-6
python -m pytest tests/test_reasoning_verifier.py                 # WS-7
python -m pytest tests/test_conversation_schema.py                # WS-8
python -m pytest tests/test_function_call.py                      # WS-9
python -m pytest tests/test_tool_use.py                           # WS-10
make eval-sft                                                      # WS-11
```

Never commit model weights or instruction shards; `ml/sft/artifacts/` and
`data/processed/<ver>/instruction/` are git-ignored. Preference tuning
(`ml/preference/`) is a follow-up plan built on the SFT checkpoints.

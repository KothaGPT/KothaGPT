# Evaluation — Implementation Plan

Goal: one harness that measures everything KothaGPT ships — knowledge,
language (Bangla + English), coding, reasoning, math, RAG, agents,
hallucination, safety, latency, and token efficiency — plus human evaluation
and a regression gate so every release (base → SFT → preference → runtime) is
measured against baselines and never silently degrades.

Guiding principles:

- Build on what exists: `evals/run.py` already runs suites (YAML: tasks →
  instances → metrics), supports pluggable targets (mock/api), and writes
  dated JSON + `REPORT.md`; `evals/metrics.py` has exact_match, ROUGE, language
  detection, Bangla script ratio, and CI helpers. This plan generalizes the
  harness to many suites/data dirs and adds the missing benchmark families.
- Every benchmark is a versioned, committed dataset (`data/benchmarks/<family>/<ver>/`)
  with train/dev/test splits and zero cross-contamination (same rule as the
  dataset pipeline).
- Every metric is defined before it is reported: no score ships without a
  documented metric + baseline. All numbers are reproducible (seeded, pinned
  target).
- Correctness first: the `mock` target (returns gold) must hit ~100% on every
  reference-scored suite — that is the harness self-check.
- Evaluation is the release gate for all other plans (SFT, preference, RAG,
  agent, runtime): their "release-grade eval" workstreams converge here.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Harness | `evals/run.py` — suite yaml → tasks → target (mock/api) → metrics → `evals/results/<date>-<name>.json` + `.REPORT.md` | hardcoded `data/benchmarks/bangla/v1`, fixed scorer map, one suite |
| Metrics | `evals/metrics.py` — exact_match, rouge, language_detection, bengali_script_ratio, mean_ci | no coding/reasoning/math/hallucination/safety/latency/token metrics |
| Bangla benchmarks | `data/benchmarks/bangla/v1` (QA 576 / translation 1120 / summarization 108 / generation 26) + `evals/suites/bangla.yaml` + reports | no test split usage, no drift baselines |
| Other suites | — | every family below is new |
| Human eval | — | no annotation tooling or judges |
| Regression | — | no baseline store, no drift gate in CI |

---

## Workstreams

### WS-0 — Eval harness generalization (`evals/run.py` refactor)

Goal: a multi-suite harness before any new benchmark.

- Refactor `evals/run.py` to: load any suite YAML with its own `data_dir`,
  `tasks`, `metrics`, and `target`; make the scorer map data-driven (per-task
  metric declarations, not hardcoded lambdas); support a `mock` self-check per
  suite; keep the dated JSON + `REPORT.md` output contract.
- Add `Target` kinds: `api` (services.api MockBackend), `engine` (runtime
  engine), `registry` (pinned model version), `human` (WS-13).
- Deliverables: generalized `run.py`, `evals/registry.py` (baseline store),
  suite schema validation.
- Metric: the existing `bangla.yaml` round-trips byte-identical reports; every
  reference-scored suite hits ~100% on `--target mock`.

### WS-1 — General knowledge benchmark (`data/benchmarks/general/`)

Goal: broad world knowledge in Bangla + English.

- Curate a general-knowledge set (facts, entities, geography, history, culture)
  with a Bangla-first question bank and English counterpart; gold short answers
  for exact-match/QA scoring.
- Deliverables: `data/benchmarks/general/v1/` + `evals/suites/general.yaml`.
- Metric: exact-match + ROUGE reported per language; baselines stored (WS-0).

### WS-2 — Bengali benchmark (`data/benchmarks/bangla/` extension)

Goal: the flagship quality gate, extended, not frozen.

- Keep v1; add v2: more QA, reading comprehension with context, grammar
  correctness, register/style, and a held-out **test** split (currently all
  `dev`). Reuse `data/pipeline` normalization for answer matching.
- Deliverables: `data/benchmarks/bangla/v2/` + `evals/suites/bangla-v2.yaml`.
- Metric: exact_match/ROUGE/language-correct per task; bengali_script_ratio
  stays at the documented floor; contamination check vs train data passes.

### WS-3 — English benchmark (`data/benchmarks/english/`)

Goal: parity English quality from the same multilingual model.

- Curate English tasks (general QA, reading comprehension, summarization,
  translation bn↔en) with gold references; keep the harness's
  language-detection metric to confirm English output.
- Deliverables: `data/benchmarks/english/v1/` + `evals/suites/english.yaml`.
- Metric: English scores reported and compared to Bangla for cross-language
  balance (multilingual plan WS-5 of `docs/sft-plan.md`).

### WS-4 — Coding benchmark (`data/benchmarks/coding/`)

Goal: measure code generation, not just prose.

- Add coding tasks: function synthesis (prompt → code), bug-fix (buggy code →
  fixed), explanation; include Bangla-comment code samples. Score with
  **execution-based** checks (unit tests on generated code) plus exact-match on
  the reference where deterministic.
- Deliverables: `data/benchmarks/coding/v1/` + `evals/suites/coding.yaml` +
  `evals/exec_runner.py` (sandboxed).
- Metric: pass@k on unit-test execution ≥ threshold; syntax validity ≥
  threshold; Bangla-comment samples handled.

### WS-5 — Reasoning benchmark (`data/benchmarks/reasoning/`)

Goal: verifiable step-by-step reasoning.

- Add reasoning tasks (logic, commonsense, multi-hop, Bangla + English) with
  gold final answers + verifier-friendly intermediate steps; score final-answer
  accuracy, not chain-of-thought prose.
- Deliverables: `data/benchmarks/reasoning/v1/` + `evals/suites/reasoning.yaml`.
- Metric: answer accuracy ≥ threshold; verifier agreement reported; no
  chain-of-thought leakage into eval scoring.

### WS-6 — Math benchmark (`data/benchmarks/math/`)

Goal: arithmetic and symbolic math, exact.

- Add math tasks: arithmetic, word problems, algebra/geometry with exact answer
  keys; normalize numeric answers (decimal/fraction tolerance) for scoring.
- Deliverables: `data/benchmarks/math/v1/` + `evals/suites/math.yaml` +
  `evals/metrics.py` numeric-normalization helpers.
- Metric: exact/numeric-tolerance accuracy ≥ threshold; Bangla word problems
  included.

### WS-7 — RAG benchmark (`data/benchmarks/rag/`)

Goal: measure retrieval → rerank → cite → answer as one pipeline.

- Add RAG tasks: retrieval-only (query → relevant doc ranking), citation
  accuracy (answer claim → cited chunk), and grounded-answer (answer supported
  by retrieved context). Consume `services/rag/` outputs via the harness target.
- Deliverables: `data/benchmarks/rag/v1/` + `evals/suites/rag.yaml` +
  recall@k / nDCG / citation-precision metrics.
- Metric: retrieval recall@k ≥ threshold; citation precision ≥ threshold;
  grounded-answer rate ≥ threshold (per `docs/rag-plan.md`).

### WS-8 — Agent benchmark (`data/benchmarks/agent/`)

Goal: end-to-end agent capability under bounded runs.

- Add agent tasks (tool selection, multi-step plan execution, failure
  recovery) with gold tool-call traces and final outcomes; drive via the agent
  loop (`docs/agent-plan.md`) and score step success + final success.
- Deliverables: `data/benchmarks/agent/v1/` + `evals/suites/agent.yaml`.
- Metric: task success rate ≥ threshold; tool-call sequence accuracy reported;
  budget (steps/cost) compliance verified.

### WS-9 — Hallucination benchmark (`data/benchmarks/hallucination/`)

Goal: groundedness as a first-class score.

- Add hallucination tasks: contextual QA with gold contexts, factual-consistency
  probes, and open-ended prompts; score NLI-style entailment of the answer in
  context plus entity/date spot-checks (aligns with `docs/preference-plan.md`
  WS-5).
- Deliverables: `data/benchmarks/hallucination/v1/` + `evals/suites/
  hallucination.yaml`.
- Metric: hallucination rate ≤ threshold; consistency score ≥ threshold on the
  suite.

### WS-10 — Safety benchmark (`data/benchmarks/safety/`)

Goal: refusal and non-toxicity as pass/fail gates.

- Add safety tasks: harmful-instruction refusal, toxic-output avoidance, PII
  non-leak, self-harm — Bangla + English, reusing `data/pipeline/toxic.py`
  blocklists as baselines; score refusal rate + benign non-refusal.
- Deliverables: `data/benchmarks/safety/v1/` + `evals/suites/safety.yaml`.
- Metric: harmful prompts refused ≥ threshold; benign prompts not refused ≤
  threshold; zero toxic/PII output (per `docs/preference-plan.md` WS-6/WS-9).

### WS-11 — Latency benchmark (`evals/benchmarks/latency/`)

Goal: serving quality numbers, not just correctness.

- Add a latency harness (distinct from the correctness harness): scripted
  prompt lengths + `--stream`, measure first-token latency, inter-token
  latency, total time, throughput (tokens/sec), and batch hit rate against a
  pinned runtime target (`docs/runtime-plan.md`); report p50/p95/p99.
- Deliverables: `evals/latency_bench.py` + `evals/suites/latency.yaml`.
- Metric: p95 first-token and tokens/sec meet the documented per-device
  budgets; numbers reproducible on the pinned runtime version.

### WS-12 — Token-efficiency benchmark (`evals/token_bench.py`)

Goal: quality per token, the economic metric.

- Add a token-efficiency harness: for each correctness task, record tokens per
  answer (via the frozen tokenizer) alongside quality, and report
  quality-per-token and compression (tokens per character); reuse the tokenizer
  efficiency gates from `docs/bangla-foundation-plan.md` (tpc, fidelity).
- Deliverables: `evals/token_bench.py` + suite.
- Metric: tokens-per-answer ≤ budget while quality ≥ threshold; unk = 0 and
  decode fidelity = 100% maintained.

### WS-13 — Human evaluation (`evals/human/`)

Goal: signal that automated metrics can't capture.

- Build an annotation workflow: side-by-side + Likert judgments (helpfulness,
  naturalness, safety, Bengali quality) with Bangla + English prompts; a judge
  UI/CLI writing judgments to `evals/human/judgments/`.
- Compute inter-annotator agreement (Cohen's κ); publish human-vs-metric
  correlation so automated gates stay honest.
- Deliverables: annotation tooling, judgment schema, agreement report.
- Metric: ≥ 2 annotators per item; κ reported; a documented correlation
  between human scores and the automated metrics used in WS-1..WS-10.

### WS-14 — Regression testing (`evals/regression.py` + CI gate)

Goal: no release ships while quality drifts.

- Add a baseline store (`evals/registry.py`, WS-0) pinning every suite's
  scores per model version; a regression runner re-runs the full suite set
  against a candidate version and fails on **statistically significant**
  degradation (mean-CI comparison via `mean_ci`) or on any hard gate (safety,
  hallucination, token gates).
- Wire `make eval-all` + `make eval-regress` into CI on every release.
- Deliverables: `regression.py`, baseline store, CI gate.
- Metric: gate fails on injected regression (tested with a degraded target);
  same-version re-run passes; reports dated and diffable.

---

## Sequencing & dependencies

```
WS-0 harness generalization
  ├──> WS-2 Bengali (flagship) ──> WS-3 English ──> WS-1 general
  ├──> WS-5 reasoning ──> WS-6 math ──> WS-4 coding
  ├──> WS-9 hallucination ──> WS-10 safety
  ├──> WS-7 RAG ──> WS-8 agent
  ├──> WS-11 latency ──> WS-12 token-efficiency
  └──> WS-13 human
WS-14 regression ──── gates every release (needs WS-0 baselines first)
```

WS-0 first — every benchmark depends on the generalized harness. WS-2/WS-3/
WS-1 are the language/knowledge spine; WS-5/WS-6/WS-4 the capability spine;
WS-9/WS-10 the hard gates; WS-7/WS-8 depend on the RAG/agent plans; WS-11/WS-12
measure the runtime; WS-13 validates the whole thing. WS-14 is the summit and
the release gate.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| General knowledge benchmark | WS-1 |
| Bengali benchmark | WS-2 |
| English benchmark | WS-3 |
| Coding benchmark | WS-4 |
| Reasoning benchmark | WS-5 |
| Math benchmark | WS-6 |
| RAG benchmark | WS-7 |
| Agent benchmark | WS-8 |
| Hallucination benchmark | WS-9 |
| Safety benchmark | WS-10 |
| Latency benchmark | WS-11 |
| Token-efficiency benchmark | WS-12 |
| Human evaluation | WS-13 |
| Regression testing | WS-14 |

## Tests

```bash
make eval-all                                            # WS-1..WS-10 correctness suites
make eval-mock                                           # harness self-check (~100% on gold)
make eval-latency                                        # WS-11
make eval-tokens                                         # WS-12
make eval-human                                          # WS-13
make eval-regress                                        # WS-14 (fails on drift)
python -m pytest tests/test_evals.py tests/test_eval_metrics.py
```

Never commit model outputs or large benchmark artifacts; `evals/results/` stays
dated and git-committed (small), `evals/human/judgments/` is git-ignored.
Every other plan's "evaluation" workstream converges on this harness.
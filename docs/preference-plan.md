# Preference / Alignment — Implementation Plan

Goal: take the instruction-tuned KothaGPT checkpoints (`ml/sft/artifacts/`) and
align them to human preference — reward model, DPO, and a release-grade
evaluation battery covering preference, hallucination, safety, helpfulness,
Bengali quality, and refusal behavior.

Guiding principles:

- Build on what exists: `ml/sft/` provides the chat template, assistant-mask
  dataset tooling, and tuned checkpoints; `ml/trainer/` provides the loop,
  checkpoint, scheduler, and monitoring. Alignment layers *preference* on top:
  pair/rejection sampling, a reward model head, DPO objective, and safety/refusal
  gates.
- Human data is precious and expensive: start with seed preference pairs from
  synthetic/rule-based judges, verify them, then hand-label a small high-quality
  set. Every record carries `{source, annotator, verified}`.
- Safety is a release gate, not an afterthought: alignment runs cannot ship
  until the safety/refusal evals (WS-6/WS-9) pass.
- Every run is reproducible: pinned to (SFT checkpoint, preference data version,
  reward model, DPO config) and recorded in `ml/preference/artifacts/<run-id>/`.
- PyTorch stays a training-only dependency; never import `torch` from the API
  service.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| SFT checkpoints | `ml/sft/` plan (`docs/sft-plan.md`); chat template + masked-loss tooling specified | SFT not yet run; no aligned-release base |
| Preference package | `ml/preference/` — `dataset.py` (`PreferenceRecord`, `load_preference_jsonl`, `split_preference`), `reward.py` (`RewardModel` head on frozen `KothaGPT`), `trainer.py` (`run_dpo`), `cli.py` (`make preference-smoke` green) | real-model RM/DPO baselines, CI gate |
| Trainer core | `ml/trainer/` loop, checkpoint (atomic/resume), scheduler, monitor, DDP/FSDP, mixed precision | only causal-LM + SFT loss paths; no reward / DPO objectives |
| Data pipeline | `data/pipeline/` gates + `data/pipeline/schema.py` (role/content) per SFT plan | no preference pair schema, no judge/annotator tooling |
| Eval | `evals/run.py` + Bangla v1, WS-4 `preference.yaml`, WS-5 `hallucination.yaml`, WS-6 `safety.yaml`, WS-7 `helpfulness.yaml`, WS-8 `bangla-quality.yaml`, WS-9 `refusal.yaml`; deterministic metrics and controls for each suite | real-model baselines, deeper linguistic quality signals, RM-guided win-rate replacing rule-based judge |
| Configs | `ml/configs/small.yaml`, `sft*.yaml` | no reward/dpo configs |

---

## Workstreams

### WS-1 — Human preference dataset তৈরি (`data/pipeline/preference.py`)

Goal: a versioned, pair-structured preference corpus (chosen vs rejected).

- Define the preference schema in `data/pipeline/preference.py`:
  `{prompt, chosen, rejected, source, annotator, task_type, quality_score}`;
  validate with the same pydantic-style gate as `schema.py`.
- Seed with **synthetic/rule-based preference pairs**: sample two SFT responses
  per prompt and rank them with a rule-based judge (length/verbosity, format
  compliance, key-fact overlap); keep only high-agreement pairs.
- Hand-label a small high-quality Bangla + English set (target ≥ 1–2k pairs)
  with inter-annotator agreement recorded; keep a held-out preference eval set
  never used in training.
- Run through Phase-B gates (toxic/PII/copyright) and B6 scoring; land shards at
  `data/processed/<ver>/preference/` with train/validation/test splits.
- Deliverables: `preference.py`, synthetic judge, hand-label manifest, shards +
  `MANIFEST.json`.
- Metric: synthetic-judge agreement with hand labels ≥ threshold; zero PII/license
  failures; held-out eval set never leaks into train (hash-checked).

### WS-2 — Reward model তৈরি (`ml/preference/reward.py`)

Goal: a scalar reward head over the frozen SFT base to score responses.

- Add a reward model head on `KothaGPT` (SFT checkpoints frozen, learn a linear
  head + optionally a final MLP over the last-token hidden state).
- Train on WS-1 pairs with a pairwise ranking loss; reuse `ml/trainer/` loop;
  record `rm.jsonl` (pairwise accuracy, rank correlation) in the run dir.
- Serve for **rejection sampling** (best-of-N response selection) — the DPO
  reference data source in WS-3.
- Deliverables: `ml/preference/reward.py`, `ml/configs/reward.yaml`, RM
  artifacts.
- Metric: held-out pairwise accuracy ≥ threshold (e.g. ≥ 70%); RM scores are
  calibrated (ties broken consistently); best-of-N beats SFT on eval.

### WS-3 — DPO pipeline তৈরি (`ml/preference/dpo.py`)

Goal: preference-tune the SFT model with Direct Preference Optimization.

- Implement the DPO loss (reference log-probs from the frozen SFT checkpoint,
  β temperature, label smoothing option) as a new trainer objective; keep the
  chat template + assistant-mask semantics from SFT.
- Generate the DPO reference data via RM-guided rejection sampling (WS-2): for
  each prompt, keep chosen = best-of-N, rejected = a mid-ranked sample.
- Reuse `ml/trainer/` loop/checkpoint/resume; `ml/configs/dpo.yaml` (β, batch,
  LR ~1e-6..5e-6, epochs, mix of preference + SFT data to prevent collapse).
- Deliverables: `ml/preference/dpo.py`, `ml/configs/dpo.yaml`, DPO artifacts.
- Metric: a CPU smoke run completes; preference eval (WS-4) improves vs SFT
  while SFT benchmarks stay non-worse; loss does not collapse to degenerate
  distributions.

### WS-4 — Preference evaluation (`evals/suites/preference.yaml`)

Goal: measure whether alignment actually follows preference.

- Extend `evals/run.py` with a preference mode: for each eval prompt,
  sample N responses from the aligned model; score with the RM (held-out RM,
  not the one used for DPO sampling) + pairwise win-rate vs the SFT baseline.
- Emit win-rate tables per task family (general, reasoning, coding,
  conversation, function-call, tool-use).
- Deliverables: `evals/suites/preference.yaml`, win-rate + RM-score report.
- Metric: aligned model wins ≥ threshold% of pairs vs SFT; per-family win-rates
  reported and no family regresses below base.

Implemented: `evals/suites/preference.yaml` + `data/benchmarks/preference/v1/preference.jsonl`
(7 triples: 3 bn, 4 en). The `preference` control in `evals/run.py` runs a
rule-based judge (`preference_judge` in `evals/metrics.py`) over
(chosen, rejected, reference); a real RM-guided win-rate replaces this baseline
before release. The suite is target-agnostic: case input is the prompt, and
chosen/rejected/reference are carried on the case. `make eval-preference`
runs it green (win rate 1.0 on the baseline triples).

### WS-5 — Hallucination evaluation (`evals/suites/hallucination.yaml`)

Goal: quantify groundedness so alignment improves factuality, not fluency.

Implemented: `evals/metrics.py` gains `semantic_entailment` (fraction of answer
content units present in context) and `semantic_contradiction` (1 - entailment),
a lexical NLI proxy over normalized unigrams/bigrams. `evals/suites/hallucination.yaml`
now reports `context_token_coverage`, `lexical_unsupported_token_rate`,
`semantic_entailment`, and `semantic_contradiction` per grounded-QA instance.
`make eval-hallucination` runs green. Remaining: broader factual-consistency and
open-ended probes, NLI-style scoring, and aligned-versus-SFT comparisons.

### WS-6 — Safety evaluation (`evals/suites/safety.yaml`)

Goal: a deterministic safety gate for release.

Implemented: `evals/suites/safety.yaml` (12 cases: 8 harmful, 4 benign) with a
`safety` control backed by `refusal_classifier`. Harmful-response inputs expect
`refuse`; benign-response inputs expect `answer`. `make eval-safety` runs green
(100% pass). Remaining: curate from `data/pipeline/toxic.py` blocklists, add
PII-leak and self-harm probes, and wire the suite as a hard release gate.

### WS-7 — Helpfulness evaluation (`evals/suites/helpfulness.yaml`)

Goal: alignment must not trade safety for uselessness.

- Prompt sets where a helpful answer is expected (benign questions, partial
  info, "I don't know" cases): score answerability, information sufficiency,
  verbosity fit, and non-refusal.
- Use a mix of reference-based (BLEU/ROUGE on gold answers where available) and
  reference-free (RM score, length-fit) signals.
- Deliverables: helpfulness suite + report.
- Metric: helpfulness score ≥ threshold; refusal-on-benign rate ≤ threshold;
  no over-verbosity regression vs SFT.

Implemented: `helpfulness_metrics` reports answerability,
reference-token information sufficiency, verbosity fit, and non-refusal, with
a composite score. `evals/suites/helpfulness.yaml` exercises useful answers,
calibrated uncertainty, non-answers, and irrelevant responses through the
`helpfulness` control (helpful score threshold: 0.75);
`make eval-helpfulness` writes the suite report. These deterministic lexical
signals are calibration baselines, not RM or human judgments.

### WS-8 — Bengali quality evaluation (`evals/suites/bangla-quality.yaml`)

Goal: guarantee alignment improves — never damages — Bangla language quality.

- Extend the Bangla v1 benchmarks (QA/translation/summarization/generation) with
  a language-quality lens: grammaticality, transliteration fidelity (via
  `ml/tokenizer/transliterate.py`), token efficiency, and register fit.
- Report aligned vs SFT vs base comparisons per metric.
- Deliverables: `evals/suites/bangla-quality.yaml` + report.
- Metric: Bengali quality metrics ≥ SFT base and ≥ documented thresholds;
  tokenizer unk rate and decode fidelity stay at 0%/100%.

Implemented: `bangla_quality_metrics` reports Bengali-script ratio, Bengali
language classification, ROUGE-L reference similarity, and a composite score.
`evals/suites/bangla-quality.yaml` checks Bengali and romanized/English
responses through the `bangla_quality` control; `make eval-bangla-quality`
writes the report (quality score threshold: 0.70). Grammar, register,
transliteration fidelity, tokenizer efficiency, and aligned-vs-SFT comparisons
remain future work. These deterministic metrics do not evaluate those areas.

### WS-9 — Refusal behavior evaluation (`evals/suites/refusal.yaml`)

Goal: calibrated refusal — refuse the harmful, answer the safe.

Implemented: `evals/suites/refusal.yaml` (8 cases: 3 harmful, 3 benign, 2
partial) with a `refusal` control backed by the three-way
`refusal_classifier` (`answer | refuse | partial`). Soft refusals that decline
+ offer an alternative are classified `partial`. `make eval-refusal` runs green
(100% pass). Remaining: confusion-style table, false-refusal/false-answer
thresholds, refusal-tone calibration across Bangla/English.

---

## Sequencing & dependencies

```text
WS-1 preference dataset ──> WS-2 reward model ──> WS-3 DPO (needs RM sampling)
                               │                        │
WS-6 safety eval ── WS-9 refusal eval ──────────────────┤  (release gates)
WS-5 hallucination ── WS-7 helpfulness ── WS-8 bangla quality ─┐
WS-4 preference eval ─────────────────────────────────────────┤
                                └──> release gate: all suites green
```

WS-1 → WS-2 → WS-3 is the critical path (dataset → reward → DPO). WS-5..WS-9
eval suites build in parallel and are the release gate for WS-2/WS-3 artifacts.
WS-4 is the primary measure of alignment quality; WS-6/WS-9 are hard gates
(no ship while harmful prompts are answered or benign prompts are refused).

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Human preference dataset তৈরি | WS-1 |
| Reward model তৈরি | WS-2 |
| DPO pipeline তৈরি | WS-3 |
| Preference evaluation | WS-4 |
| Hallucination evaluation | WS-5 |
| Safety evaluation | WS-6 |
| Helpfulness evaluation | WS-7 |
| Bengali quality evaluation | WS-8 |
| Refusal behavior evaluation | WS-9 |

## Tests

```bash
python -m pytest tests/test_preference_schema.py                # WS-1
make rm-smoke && python -m pytest tests/test_reward.py           # WS-2
make dpo-smoke && python -m pytest tests/test_dpo.py             # WS-3
make eval-preference                                             # WS-4
make eval-hallucination                                          # WS-5
make eval-safety                                                 # WS-6
make eval-helpfulness                                            # WS-7
make eval-bangla-quality                                         # WS-8
make eval-refusal                                                # WS-9
make eval-alignment                                              # WS-4..WS-9
```

Never commit model weights or preference shards; `ml/preference/artifacts/` and
`data/processed/<ver>/preference/` are git-ignored. Alignment builds directly on
`docs/sft-plan.md` checkpoints; preference tuning is upstream of the safety
release checklist.
# Foundation — Implementation Plan

Goal: formally settle the product decisions the whole KothaGPT stack builds on
— the AI's core purpose, language scope, product type, target users, MVP
feature list, model size, license/open-source strategy, and project identity —
so every downstream plan has fixed, documented premises instead of implicit
assumptions.

Guiding principles:

- Decide in writing, then freeze: each decision lands in a `DECISION.md` /
  `docs/foundation.md` record with rationale and rejected alternatives (same
  style as `ml/models/DECISION.md`), so later plans stop re-litigating scope.
- Decisions are hypotheses with owners and review dates: language scope, model
  size, and license can be revisited with evidence, but only through the
  recorded decision process.
- The MVP is a slice of the existing plans, not a new one: it selects a minimal
  path through `docs/dataset-pipeline-plan.md` → `docs/base-model-plan.md` →
  `docs/pretraining-plan.md` → `docs/runtime-plan.md` and defines the first
  user-visible milestone.
- The repository is the source of truth: project naming, ownership, and CI
  standards are settled here and reflected in `README.md`, `TODO`, and
  `.github/`.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Project name / repo | "KothaGPT", repo bootstrapped, `README.md`, `docs/roadmap.md`, CI, Makefile, config system | no written naming/ownership rationale |
| Language scope | Bangla-first multilingual in practice (`docs/bangla-foundation-plan.md`, dataset plan A2 English/mixed) | no formal decision record with budget/timeline |
| Product type | chat + coding + agents + research all implied across plans | no prioritization / sequencing decision |
| Target users | developer-focused in practice (SDK-first, `docs/sdk.md`, `docs/ecosystem-plan.md`) | no persona definitions or validation |
| MVP | `docs/web-app-plan.md` MVP slice (dashboard, chat, models, playground, API keys, settings) | no end-to-end MVP timeline with acceptance criteria |
| Model size | `ml/configs/small.yaml` (hidden 768/12L) + `large.yaml` planned in pretraining WS-12 | no explicit 1B/3B/7B/14B+ target decision |
| License | `docs/ecosystem-plan.md` WS-10 (OSS code, weights separate) | no license record, no chosen SPDX identifiers |
| Security/IR | — | foundation record must name the security owner (see `docs/security-plan.md`) |

---

## Workstreams

### WS-1 — AI-এর মূল উদ্দেশ্য নির্ধারণ (core purpose)

Goal: a one-paragraph mission every plan can be checked against.

- Write the mission + non-goals (what KothaGPT will *not* do in v1) and the
  success definition (from `TODO` metrics: data coverage, eval scores, latency,
  cost).
- **Decision record** `docs/foundation.md` with rationale + review date.
- Deliverables: mission statement, non-goals, success metrics.
- Metric: every downstream plan references the same mission line; non-goals are
  traceable (nothing in scope contradicts them).

### WS-2 — Language scope ঠিক করা (bn / en / multilingual)

Goal: a fixed language budget.

- Decide the scope: **Bangla-first bilingual (bn+en) v1**, with a documented
  multilingual (add `hi` etc.) trigger criteria + timeline.
- Record the corpus implications (dataset plan A1/A2 language gates) and eval
  implications (eval plan WS-2/WS-3).
- Deliverables: scope decision, language budget (data/token/eval share), review
  trigger.
- Metric: the chosen share is reflected in dataset pipeline gates and eval
  suites; a trigger (e.g. bn benchmarks ≥ threshold) fires the multilingual
  extension.

### WS-3 — Product type নির্বাচন (chat / coding / agent / research)

Goal: pick the primary product surface, sequence the rest.

- Choose **chat-first** as the primary (web-app Sprint 2 is the first release),
  then coding (sft WS-6), then agents (agent plan), research tooling (RAG plan)
  — with a written rationale and dependency map.
- Record what each type needs upstream (SFT data, function-calling, tool
  registry, retrieval) so plans align.
- Deliverables: product-type decision + roadmap alignment.
- Metric: the MVP release ships the primary surface; secondary surfaces are
  explicitly sequenced, not implicit.

### WS-4 — Target users নির্ধারণ

Goal: defined personas the UX and docs optimize for.

- Define 2–3 personas (Bangla-speaking developer, bilingual student/creator,
  enterprise team) with jobs-to-be-done, and map them to the platform
  (`docs/web-app-plan.md`), SDK (`docs/sdk.md`), and ecosystem
  (`docs/ecosystem-plan.md`) surfaces.
- Persona validation via the human-eval workflow (`docs/eval-plan.md` WS-13).
- Deliverables: persona docs, JTBD statements.
- Metric: each persona has a documented happy path through the product; UX
  decisions reference personas.

### WS-5 — MVP feature list তৈরি

Goal: a shippable first release with acceptance criteria.

- Enumerate the MVP (from `docs/web-app-plan.md` MVP): dashboard, streaming
  chat, model selector, playground, API keys, settings — each with an
  acceptance test (from `docs/eval-plan.md` + API tests).
- Define the MVP exit criteria: p95 latency budget, chat quality floor, key
  management correctness.
- Deliverables: MVP list + exit checklist in `docs/foundation.md`.
- Metric: MVP exit checklist is all green in CI before release.

### WS-6 — Model size নির্ধারণ (1B / 3B / 7B / 14B+)

Goal: a justified model-size target with a path.

- Decide the target: **1B-class v1** (CPU/GPU-friendly, matches the 16k frozen
  tokenizer and `small.yaml`-scale budgets) with documented headroom to
  3B/7B via pretraining WS-11/WS-12 scale-out.
- Record the constraint math: data volume vs params (Chinchilla-style),
  memory per device, training cost, latency budget from `docs/runtime-plan.md`.
- Deliverables: size decision + scaling plan.
- Metric: the v1 config rounds-trips on the chosen devices; the scaling plan
  lists the data/GPU/EVB needed for each step up.

### WS-7 — License ও open-source strategy নির্ধারণ

Goal: clear, compatible licenses for code vs weights vs data.

- Decide: **code MIT/Apache-2.0**, **model weights under a separate
  permissive-with-attribution license**, datasets under their source licenses
  (dataset plan B5 strict allow-list).
- Record SPDX identifiers in a `LICENSE*` / `NOTICE` set and the contribution
  process (ecosystem WS-10).
- Deliverables: license decision, LICENSE/NOTICE files, SBOM gate.
- Metric: CI SBOM/license scan is green; every dataset carries a validated
  license; the weights license is distinct and documented.

### WS-8 — Project name ও repository তৈরি

Goal: stable identity and repository hygiene.

- Fix the name/branding ("KothaGPT") and ownership map (from `TODO` owners);
  settle repo conventions (branching, PR, issue labels, CODEOWNERS) and the
  `Makefile`/CI standard.
- Ensure `README.md` and `TODO` reflect the decisions above.
- Deliverables: branding + governance docs, repo config.
- Metric: CI runs on every PR; CODEOWNERS routes plans/SDK/API correctly;
  README links the foundation record.

---

## Sequencing & dependencies

```
WS-1 purpose ──> WS-2 language ──> WS-3 product type ──> WS-5 MVP
WS-4 personas (parallel with WS-1..3) ──> WS-5
WS-6 model size (needs WS-2 data budget) ──> WS-7 license
WS-8 identity/governance (wraps everything)
```

WS-1 → WS-2 → WS-3 is the decision spine; WS-4 informs UX; WS-5 is the
shipment point; WS-6/WS-7 fix constraints; WS-8 codifies it all.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| AI-এর মূল উদ্দেশ্য নির্ধারণ | WS-1 |
| বাংলা, ইংরেজি নাকি multilingual AI হবে ঠিক করা | WS-2 |
| Chat AI / Coding AI / Agent AI / Research AI নির্বাচন | WS-3 |
| Target users নির্ধারণ | WS-4 |
| MVP feature list তৈরি | WS-5 |
| Model size নির্ধারণ — 1B / 3B / 7B / 14B+ | WS-6 |
| License ও open-source strategy নির্ধারণ | WS-7 |
| Project name ও repository তৈরি | WS-8 |

## Tests

```bash
make plans-check                                   # structure + links
python -m pytest tests/test_foundation_contract.py # each decision record exists + is referenced
```

No runtime code ships from this plan; its "tests" are the decision records
being written, referenced by downstream plans, and passing the plan checker.
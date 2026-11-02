# Foundation Decision Record

Living record of the product decisions fixed by `docs/foundation-plan.md`.
Each entry states the decision, the rationale, and rejected alternatives.
Status: **TENTATIVE** until confirmed by the foundation workstreams; change any
entry only by updating the record and re-running the plan checker.

## WS-1 — Core purpose (AI-এর মূল উদ্দেশ্য)

- **Decision:** Bangla-first, developer-focused AI platform covering the full
  stack (tokenizer → dataset → base model → instruction → preference → runtime
  → RAG → agents → SDK → platform → hub).
- **Rationale:** aligned with `README.md`, `TODO`, and every plan in `docs/`.
- **Rejected:** single-model API service with no own stack.

## WS-2 — Language scope (bn / en / multilingual)

- **Decision:** Bangla-first **bilingual bn+en for v1**; multilingual extension
  (e.g. `hi`) gated behind a trigger (Bangla benchmarks ≥ threshold).
- **Rationale:** frozen 16k Bangla BPE tokenizer, Bangla-first corpus, bn+en
  eval suites.
- **Rejected:** multilingual from day one (data/token budget too thin).

## WS-3 — Product type (chat / coding / agent / research)

- **Decision:** **Chat-first** primary surface (web-app Sprint 2), then coding
  (SFT), then agents, then research/RAG.
- **Rationale:** fastest user-visible milestone; others depend on upstream data
  + tooling.

## WS-4 — Target users

- **Decision (proposed):** Bangla-speaking developer, bilingual student/creator,
  enterprise team (to be validated via eval-plan WS-13 human eval).

## WS-5 — MVP

- **Decision:** dashboard, streaming chat, model selector, playground, API keys,
  settings (from `docs/web-app-plan.md` MVP), with p95 latency + chat quality
  exit criteria.

## WS-6 — Model size

- **Decision (proposed):** **1B-class v1** on the 16k frozen tokenizer, with a
  documented scaling path to 3B/7B (pretraining WS-11/WS-12).
- **Rationale:** CPU/GPU-friendly, matches small.yaml-scale budgets and
  per-device latency budgets.

## WS-7 — License & open-source

- **Decision (proposed):** code MIT/Apache-2.0; model weights under a separate
  permissive-with-attribution license; datasets keep their source licenses
  (dataset-plan B5 allow-list); SBOM + license scan in CI.

## WS-8 — Project identity

- **Decision:** name "KothaGPT", repo bootstrapped, CI + Makefile + config
  system in place; CODEOWNERS routing to be finalized.
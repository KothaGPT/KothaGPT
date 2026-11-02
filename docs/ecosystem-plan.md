# KothaGPT AI Ecosystem — Implementation Plan

Goal: wrap the entire platform in a user-facing ecosystem — model hub, dataset
hub, agent/tool marketplaces, prompt library, evaluation hub, developer portal,
playground, community, open-source repositories, a managed fine-tuning
platform, hosted inference, enterprise API, and local/offline AI — all layered
on the plans below it (`docs/web-app-plan.md`, `docs/sdk.md`, `docs/infra-plan.md`).

Guiding principles:

- Build on what exists: `apps/web` (Next.js dashboard, per `docs/web-app-plan.md`),
  `apps/playground`, `apps/docs`, the multi-language SDKs (`packages/*`,
  `docs/sdk.md`), the model registry (`docs/runtime-plan.md`), dataset versioning
  (`docs/dataset-pipeline-plan.md`), and the eval harness (`docs/eval-plan.md`).
  The ecosystem is *surfaces*, not new plumbing: hubs list registry entries,
  marketplaces publish registry catalogs, the portal wraps existing APIs.
- Hubs are registries with a face: every hub (models, datasets, agents, tools,
  prompts, evaluations) is a read/write registry + UI + publish/version
  workflow, reusing the content-addressed versioning idea everywhere.
- Marketplace trust is non-negotiable: every published item carries
  `{owner, license, provenance, verified}`; nothing untrusted executes until it
  passes review (agents/tools) or the license/PII gates (datasets).
- The web app boundary holds: UI → typed API client → API Gateway → services.
  No new backend feature is exposed to the UI without going through `/v1/*`.
- Local/offline is a first-class tier, not an afterthought: the same model
  runs on a laptop without cloud, and the ecosystem degrades gracefully offline.
- PyTorch never crosses into the web/API layer; inference/fine-tuning stay in
  their services behind the gateway.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Web app | `apps/web` scaffold + full plan (`docs/web-app-plan.md`): dashboard, chat, models, datasets, training, eval, RAG, agents, playground, API keys, usage | not built yet; hubs/marketplaces/community not in scope of the web-app plan |
| Playground | `apps/playground` exists | no public playground surface |
| Docs | `apps/docs` exists | no developer portal with keys/onboarding |
| SDKs | `packages/*` (python, ts, go, rust, cli, tools, agents) + `docs/sdk.md` + `examples/` | no marketplace/client plugin flow, no local/offline mode |
| Registry | model registry planned (`docs/runtime-plan.md` WS-9/WS-10) | no public model hub; no dataset/agent/tool/prompt registries |
| Eval | `docs/eval-plan.md` harness + `evals/results/` | no evaluation hub / leaderboard |
| Fine-tuning | SFT/DPO plans (`docs/sft-plan.md`, `docs/preference-plan.md`) | no managed/self-serve fine-tuning platform |
| Infra | `docs/infra-plan.md` (GPU cluster, serving, DR) | no tenant/billing/enterprise tier; no offline packaging |

---

## Workstreams

### WS-1 — AI model hub (`apps/web` + `services/hub/`)

Goal: browse, publish, download, and serve models like a registry storefront.

- Extend the model registry (`docs/runtime-plan.md` WS-9/WS-10) into a public
  hub: model cards (description, license, metrics, quant options, served
  status), publish/version/rollback, download (digest-pinned), and "serve on
  hosted inference" (WS-12).
- Hubs are read/write registries: published model versions are immutable;
  `stable`/`latest` aliases shown in the hub.
- Deliverables: `services/hub/` + hub routes in `apps/web` (models page,
  publish flow, model cards).
- Metric: publish → browse → download round-trips with digest verification;
  every card shows license + provenance; hosted-serving toggle works.

### WS-2 — Dataset hub (`services/hub/datasets` + `apps/web` datasets page)

Goal: versioned, gated public datasets (reuse the data pipeline).

- Publish datasets as versions (content-addressed, per `docs/dataset-pipeline-plan.md`
  C2) with stats (token count, language mix, dedup rate, quality histogram,
  license, PII status) and download/preview.
- Upload flow runs the Phase-B gates (license/PII/toxic) before a version is
  published; provenance is mandatory.
- Deliverables: dataset hub API + UI, publish workflow.
- Metric: a dataset publishes only after gates pass; version history +
  rollback works; preview matches the shard contents.

### WS-3 — Agent marketplace (`services/marketplace/agents`)

Goal: publish and install agents with a trust gate.

- Registry of agent packages (spec + manifest, per `docs/agent-plan.md`): owner,
  version, permission scope, sandbox profile, verified status, review state.
- Marketplace UI (browse, install, version) in `apps/web`; installed agents
  run through the agent loop with the manifest's permission scope enforced
  (no privilege escalation on install).
- Deliverables: agent marketplace API + UI, review workflow.
- Metric: install enforces the manifest's permissions; reviewed agents are
  marked `verified`; rollback to a prior agent version works.

### WS-4 — Tool marketplace (`services/marketplace/tools`)

Goal: third-party tools that execute only inside the sandbox.

- Publish tools against the tool registry spec (`docs/agent-plan.md` WS-1)
  with explicit permission + sandbox requirements; review gate before
  `enabled`.
- Marketplace UI (browse, enable per-agent, versions) in `apps/web`.
- Deliverables: tool marketplace API + UI, review gate.
- Metric: unreviewed tools never become enabled; permissions are explicit and
  audited; a tool's invocation is sandbox-confined (agent plan WS-15).

### WS-5 — Prompt library (`apps/web` prompts page + `services/hub/prompts`)

Goal: a curated, shareable library of prompt templates.

- Prompt registry (title, task, template with `{placeholders}`, language,
  model hints, tags, community copies); CRUD + search + clone; a "run in
  playground" (WS-8) action.
- Deliverables: prompt API + UI + seed Bangla/English library.
- Metric: prompt CRUD + search round-trips; placeholders validate; clone →
  run-in-playground works.

### WS-6 — Evaluation hub (`apps/web` evaluations page + leaderboard)

Goal: every model's scores are public, comparable, and regression-gated.

- Surface `docs/eval-plan.md` results: per-family score cards, model vs model
  comparison, and a public leaderboard fed by the regression gate's baseline
  store; mark `draft` vs `published` reports.
- Deliverables: eval hub API (over `evals/registry.py`) + UI + leaderboard.
- Metric: a published report renders scores + CIs; comparison table matches the
  baseline store; leaderboard only shows `published` versions.

### WS-7 — Developer portal (`apps/docs` + `apps/web` developer section)

Goal: onboarding that turns a reader into a shipping integrator.

- Docs (per `docs/sdk.md`): quickstarts per language, API reference, examples;
  API-key management (create/revoke/scopes/limits/usage, one-time full-key
  display) and SDK install snippets with copy-paste.
- Interactive auth flow: sign-up → key → first request in ≤ 5 minutes.
- Deliverables: portal routes in `apps/docs`/`apps/web`, key management UI.
- Metric: time-to-first-request measured; docs build + lint clean; key
  revocation works immediately.

### WS-8 — AI playground (`apps/playground`)

Goal: a public, no-signup playground that showcases the stack.

- Chat playground (model selector, params, tools, RAG/KB toggle, streaming,
  JSON mode, token counter, latency) backed by the API gateway; "compare two
  models" and "export API code" (SDK snippets) actions.
- Public tier has rate limits (infra plan WS-4 / runtime WS-13); signed-in users
  get persisted playground sessions.
- Deliverables: `apps/playground` build-out, export-code feature.
- Metric: playground chat streams end-to-end; export snippets run verbatim;
  public tier respects rate limits without 401 noise.

### WS-9 — Community (`apps/web` community + forums)

Goal: a feedback loop, not a content silo.

- Community surfaces: prompt sharing (from WS-5), model/dataset discussions,
  feedback and issue links, changelog, and moderation basics (report/remove).
- Tie community votes to hub "trending" while keeping trust gates separate
  (votes never override verified status).
- Deliverables: community routes, moderation, changelog.
- Metric: share/comment/report round-trips; moderation removes flagged items;
  community engagement metrics reported.

### WS-10 — Open-source repositories

Goal: everything that can be OSS is OSS, with a clean release process.

- Codify the OSS strategy: MIT/Apache where possible, `CODE_OF_CONDUCT`,
  `CONTRIBUTING`, issue/PR templates, CI on every PR (already exists in
  `.github/workflows/ci.yml`), release tags + changelog, and SBOM/license
  scanning in CI.
- Publish SDKs + CLI to registries (PyPI/npm/crates.io/Go proxy) with signed,
  tagged releases; keep the base model weights licensed separately from code.
- Deliverables: OSS governance docs, release automation, SBOM CI gate.
- Metric: PRs pass CI + lint; releases are tagged with changelogs; SBOM scan
  clean; SDK releases installable from public registries.

### WS-11 — Model fine-tuning platform (`services/fine-tune/`)

Goal: self-serve fine-tuning (SFT + preference) on hosted GPUs.

- Job API + UI wrapping the SFT/DPO pipelines (`docs/sft-plan.md`,
  `docs/preference-plan.md`): pick base model + dataset (own or hub WS-2) +
  hyperparameters; job runs on the GPU cluster (infra WS-1) with live loss/LR
  graphs and checkpoint download; gates results through the eval hub (WS-6)
  before activation.
- Quotas/billing per tenant; job results are digest-pinned and versioned.
- Deliverables: `services/fine-tune/`, training UI (web-app Sprint 5), job
  queue wiring.
- Metric: a hosted SFT job completes, passes eval gates, and activates a new
  hub model version; job failure resumes/restarts cleanly.

### WS-12 — Hosted inference

Goal: managed serving with usage-based pricing.

- Route hosted requests to the serving stack (`docs/infra-plan.md` WS-2..WS-4)
  per tenant with key-based auth (runtime WS-12), rate limits (runtime WS-13),
  usage metering (tokens in/out per key/tenant), and billing integration.
- Model availability = hub `served` status; a served version is always the
  registry-pinned one.
- Deliverables: metering + billing hooks, `served` workflow, SLA dashboards.
- Metric: usage metering matches the token-efficiency numbers; billing
  round-trips; a `served` model is always the registry-pinned version.

### WS-13 — Enterprise API

Goal: the same stack with enterprise guarantees.

- Enterprise tier: SSO (SAML/OIDC), audit logs (who/when/what per API call),
  dedicated capacity (isolated GPU pool or cluster), data residency options,
  VPC peering, contractual SLA backed by the SLOs in `docs/infra-plan.md`.
- Add enterprise config surface (security settings, retention, admin roles).
- Deliverables: SSO + audit + dedicated capacity, enterprise admin UI.
- Metric: SSO login round-trips; audit log captures all admin + data actions;
  dedicated capacity isolates tenants (tested); SLA reports generated.

### WS-14 — Local/offline AI (`packages/local` + `apps` offline mode)

Goal: KothaGPT runs without the cloud.

- Package the runtime (`docs/runtime-plan.md`) for local: a self-contained
  offline bundle (model weights quantized for laptop/desktop, local KV cache,
  CPU inference path from WS-6) exposed via a local gateway; CLI + SDKs point
  at the local endpoint via config.
- Offline mode in `apps/playground`/`apps/web`: feature-detect connectivity,
  degrade gracefully (no hubs/marketplace offline; local models + prompts +
  KB still work).
- Deliverables: offline bundle + installer, local gateway, offline SDK mode.
- Metric: offline bundle serves chat on a reference laptop within the CPU
  latency budget; SDKs switch local↔cloud by config; offline UI hides
  unavailable features cleanly.

---

## Sequencing & dependencies

```
WS-1 model hub ──> WS-6 eval hub ──> WS-7 developer portal ──> WS-8 playground
WS-2 dataset hub ──> WS-11 fine-tuning (feeds new hub models) ──> WS-12 hosted inference
WS-3 agent marketplace ──> WS-4 tool marketplace ──> WS-5 prompt library
WS-9 community (wraps WS-1..WS-5 sharing)
WS-10 open source (foundation, runs throughout)
WS-13 enterprise (on top of WS-12)
WS-14 local/offline (parallel tier)
```

WS-1/WS-2 (registries → hubs) are the foundation — every other surface wraps
them. WS-6 and WS-7 make the platform trustworthy and usable; WS-8 is the
showcase. WS-3/WS-4/WS-5 build on the agent/tool registries. WS-11 feeds WS-1
and needs WS-12; WS-13 layers guarantees on WS-12; WS-14 is an independent
parallel tier. WS-10 governs the whole thing.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| AI model hub | WS-1 |
| Dataset hub | WS-2 |
| Agent marketplace | WS-3 |
| Tool marketplace | WS-4 |
| Prompt library | WS-5 |
| Evaluation hub | WS-6 |
| Developer portal | WS-7 |
| AI playground | WS-8 |
| Community | WS-9 |
| Open-source repositories | WS-10 |
| Model fine-tuning platform | WS-11 |
| Hosted inference | WS-12 |
| Enterprise API | WS-13 |
| Local/offline AI | WS-14 |

## Tests

```bash
python -m pytest tests/test_hub.py tests/test_marketplace.py   # WS-1..5
python -m pytest tests/test_eval_hub.py                        # WS-6
cd apps/docs && pnpm lint && pnpm build                        # WS-7
cd apps/playground && pnpm lint && pnpm build                  # WS-8
python -m pytest tests/test_community.py                       # WS-9
make oss-verify                                               # WS-10 (CI + SBOM)
python -m pytest tests/test_fine_tune.py                       # WS-11
python -m pytest tests/test_metering.py tests/test_billing.py  # WS-12
python -m pytest tests/test_enterprise_api.py                  # WS-13
python -m pytest tests/test_local_runtime.py                   # WS-14
make eval-regress                                             # ecosystem-wide gate
```

Never commit keys, weights, or user content; hub/marketplace tenant data and
local bundles are git-ignored. The ecosystem is the top layer of every plan in
`docs/`: it consumes the runtime, RAG, agents, eval, and infra plans, and the
web-app plan (`docs/web-app-plan.md`) + SDKs (`docs/sdk.md`) are its UI.
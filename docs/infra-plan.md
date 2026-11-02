# Production Infrastructure — Implementation Plan

Goal: a deployable, observable, recoverable production platform for the
KothaGPT stack — GPU cluster + model serving with load balancing and
autoscaling, backed by Redis caching, PostgreSQL, object storage, a vector
database and a queue system — with full observability (metrics, logs, tracing),
cost monitoring, and disaster recovery, codified in IaC (`infra/terraform/`,
`infra/kubernetes/`, `infra/docker/`).

Guiding principles:

- Build on what exists: `docker-compose.yml` already runs Postgres, Redis,
  Qdrant, and the API; `infra/README.md` already lists the intent (GPU envs,
  model registry, object storage, K8s, observability, CI/CD, secrets); the CI
  gate exists in `.github/workflows/ci.yml`. This plan turns those intentions
  into reproducible infrastructure.
- Everything is code: Terraform (cloud), Kubernetes (deploy), Docker (image),
  and a documented secrets strategy. No manual server state is production.
- Environments are mirrors: staging is byte-for-byte the same IaC as prod with
  smaller scale, so a prod issue is reproducible in staging first.
- Observability is a product requirement: every service emits metrics/logs/
  traces by default; no service is deployable without them (enforced by a
  template/CI check).
- Data is durable and restorable: PostgreSQL, object storage, vector data, and
  the model registry all have tested backup + restore (DR) — "it's backed up"
  only counts after a successful restore drill.
- Secrets never touch the repo: all credentials come from a vault/secret
  manager at deploy time.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Local stack | `docker-compose.yml` (postgres, redis, qdrant, api) | no GPU profile, no inference/agent/rag services, no observability, no secrets |
| Infra intent | `infra/README.md` lists GPU envs, registry, object storage, K8s, observability, CI/CD, secrets | `infra/docker/` README stub; `infra/kubernetes/`, `infra/terraform/` empty |
| CI/CD | `.github/workflows/ci.yml` (data + tokenizer + tests) | no deploy pipeline, no eval gate on release, no image build/push |
| Model serving | runtime plan defines engine + registry (`docs/runtime-plan.md`) | no serving deployment, no LB, no autoscaling |
| Data services | compose postgres/redis/qdrant (dev) | no managed/prod equivalents, no backups, no caching layer wiring |
| Observability | none | metrics/logs/tracing/cost monitoring all missing |
| Secrets | `.env*` files present (dev) | no vault, no rotation, no audit |

---

## Workstreams

### WS-1 — GPU cluster (`infra/terraform/gpu/`)

Goal: a reproducible, schedulable pool of GPU nodes.

- Terraform for the GPU cluster: node groups with pinned GPU type/count,
  driver + container-runtime bootstrap, spot/on-demand mix policy, and node
  labels/taints for scheduling (training vs serving).
- Add `make cluster-up` / `make cluster-down` with a state lock (S3/DynamoDB)
  and a `cluster-plan` dry-run.
- Deliverables: Terraform modules, `infra/README.md` cluster section.
- Metric: `terraform plan` applies cleanly from a clean state; a GPU node
  passes the runtime `gpu-verify` smoke (`docs/pretraining-plan.md` WS-4);
  teardown leaves no orphaned resources.

### WS-2 — Model serving (`infra/kubernetes/model-serving/`)

Goal: the runtime engine deployed as a versioned, scalable service.

- Kubernetes manifests for the inference service (from `docs/runtime-plan.md`):
  Deployment (pinned image + model version + quant profile), Service, HPA
  wiring (WS-4), and a readiness probe that runs a real one-shot generation.
- Rolling updates with zero downtime; a failed model version rolls back
  automatically (registry `failed` status gated).
- Deliverables: manifests + `make deploy-serving`.
- Metric: a version rollout is zero-downtime; readiness fails closed on a bad
  version; rollback re-points to the last good version in ≤ documented time.

### WS-3 — Load balancing (`infra/kubernetes/ingress/`)

Goal: distribute traffic across serving replicas correctly.

- Ingress + service mesh (or LB controller) with session affinity for chat/
  streaming, connection draining on scale-down, and per-route timeouts
  matching the API limits.
- Health-aware routing: only `ready` registry versions receive traffic.
- Deliverables: ingress manifests, LB config.
- Metric: streaming connections are stable across replica scale events; a
  failing replica is removed from rotation within the health-check interval;
  no traffic to non-ready versions.

### WS-4 — Autoscaling (`infra/kubernetes/autoscale/`)

Goal: scale with demand, without thrash.

- HPA on the serving service keyed to custom metrics (queue depth, tokens/sec,
  GPU utilization, request rate) with min/max bounds from the runtime budget
  (`docs/runtime-plan.md`); scale-down cooldown tuned to streaming connection
  drain.
- A load-shedding path (429 via rate limit) when the hard ceiling is hit.
- Deliverables: HPA manifests + custom metric adapter.
- Metric: load spike scales replicas within the latency target; scale-down
  doesn't drop in-flight streams; no thrash (scale events ≤ N/hour).

### WS-5 — Redis caching (`infra/redis/` + `services/api` cache)

Goal: take load off the engine, not just off the DB.

- Promote Redis from dev container to managed/prod cluster; wire cache layers:
  KV + prompt-cache hit metadata, chat response caching for idempotent
  requests (TTL, never cache streaming or sensitive data), and rate-limit
  counters (runtime plan WS-13) backed by it.
- Cache-aware backends: embedding/rerank results cached by input hash.
- Deliverables: Redis cluster config, cache client, tests.
- Metric: cache hit rate reported per endpoint; correctness verified (cached
  reply == fresh reply); no sensitive/session data in cache.

### WS-6 — PostgreSQL (`infra/postgres/`)

Goal: a managed, backed-up relational store for registry/KB/memory/tenants.

- Move from the dev container to a managed Postgres (or operator-run) with PITR,
  connection pooling (PgBouncer-style), TLS, and per-service roles with least
  privilege.
- Schema migrations via a versioned migration tool; schema version pinned in
  the registry.
- Deliverables: Postgres IaC, migration tooling, roles.
- Metric: migrations apply cleanly on empty + current DBs; backup/restore
  drill passes (WS-15); services connect only with their own role.

### WS-7 — Object storage (`infra/object-storage/`)

Goal: durable, cheap storage for weights, datasets, snapshots, uploads.

- S3-compatible bucket layout: `models/` (registry artifacts), `datasets/`,
  `backups/`, `uploads/`; versioning + lifecycle (transition to cheaper tiers,
  expiry), server-side encryption, and bucket policies per service.
- Model/dataset artifacts upload from CI and are pulled at deploy (registry
  digest-pinned).
- Deliverables: bucket IaC, artifact upload/pull tooling.
- Metric: artifact upload → pull round-trips with digest verification; bucket
  policies deny cross-tenant access; lifecycle rules tested.

### WS-8 — Vector database (`infra/qdrant/`)

Goal: Qdrant as a managed, backed-up service (RAG's store).

- Promote Qdrant from dev container to a production deployment with
  persistence volumes, snapshots to object storage (WS-7), and per-KB
  collection lifecycle (rag plan WS-12).
- Deliverables: Qdrant IaC + snapshot/restore wiring.
- Metric: collection snapshot → restore round-trips; retrieval quality unchanged
  after restore; backup cadence verified.

### WS-9 — Queue system (`infra/queue/`)

Goal: async work (ingestion, embedding, agent jobs) without blocking HTTP.

- Add a queue (Redis Streams or a dedicated broker) for async jobs: document
  ingestion (`docs/rag-plan.md` WS-1), embedding batches, agent runs
  (long tasks), and notification hooks.
- Worker deployment with retries, dead-letter queue, and per-queue concurrency
  limits.
- Deliverables: queue config + worker manifests.
- Metric: jobs survive worker restarts; DLQ captures failures with context;
  embedding/ingestion throughput scales with workers.

### WS-10 — Observability (`infra/observability/`)

Goal: one place to see every service.

- Deploy the observability stack (metrics + logs + traces below): Prometheus +
  Grafana, Loki (or equivalent logs), and Tempo/Jaeger (traces), with an
  OpenTelemetry collector as the single ingestion point.
- Every service exposes health/readiness + metrics by default (CI-enforced).
- Deliverables: observability manifests, OTel collector config, dashboards.
- Metric: all services report metrics/logs/traces (CI check); dashboards show
  the runtime/API/RAG/agent panels; alert rules fire on SLO breach.

### WS-11 — Metrics (`services/*` metric hooks + Prometheus)

Goal: numbers that drive scaling and budgets.

- Export service metrics: request rate/latency (p50/p95/p99), error rate,
  tokens/sec, GPU util, cache hit rate, queue depth, model version in service.
- Prometheus scrape config + alerting rules (SLO-based: latency, error budget,
  saturation).
- Deliverables: metric exporters, Prometheus + alert rules.
- Metric: latency/error SLOs are computed and alerted; the runtime's latency
  benchmark (`docs/eval-plan.md` WS-11) numbers match live metrics.

### WS-12 — Logs (`infra/observability/logs`)

Goal: searchable, structured logs with context.

- Structured JSON logs from every service (request id, trace id, model
  version, tenant) via the runtime plan WS-11 wiring; shipped to the log store
  with retention tiers.
- Log redaction: PII/secrets masked at emission, never logged in plaintext.
- Deliverables: structured-logging convention, log shipping, retention policy.
- Metric: every request has a greppable trace-id trail; redaction verified on
  PII fixtures; retention honored.

### WS-13 — Tracing (`infra/observability/tracing`)

Goal: end-to-end request paths, not isolated metrics.

- OTel traces across the edges: API → engine (KV/batch) → RAG → agents; spans
  carry model version, quant level, tool calls, and chunk sources.
- Trace sampling (head-based with error-priority) to bound cost while keeping
  every error trace.
- Deliverables: OTel instrumentation, trace service, sampling config.
- Metric: a chat request produces one complete trace across services; sampled
  error traces are always retained; p95 latency breaks down by span.

### WS-14 — Cost monitoring (`infra/cost/`)

Goal: know what every component costs before the invoice.

- Tag all IaC resources (service, environment, tenant); daily cost reports per
  tag; per-model/per-endpoint token cost attribution (via the runtime + eval
  token-efficiency numbers).
- Budget alerts on projected spend; a documented cost/SLO trade-off table.
- Deliverables: cost tags, reporting dashboard, budget alerts.
- Metric: per-service and per-model cost is reportable weekly; budget alert
  fires on forecast breach; cost-per-token tracked.

### WS-15 — Disaster recovery (`infra/dr/`)

Goal: restorable everything, tested on a schedule.

- Backup + restore matrix: Postgres (PITR), object storage (versioning +
  replication), Qdrant snapshots, Redis (persistence), registry state; every
  service documents an RPO/RTO.
- DR drill automation: `make dr-drill` restores a full stack from backups into
  a clean environment and runs the eval/regression gate against it.
- Deliverables: backup IaC, DR runbook, drill automation.
- Metric: a quarterly drill restores the stack within RTO and the regression
  gate (`docs/eval-plan.md` WS-14) passes against the restored data; RPO
  verified by recoverable-backup timestamps.

---

## Sequencing & dependencies

```
WS-1 GPU cluster ──> WS-2 model serving ──> WS-3 LB ──> WS-4 autoscaling
WS-5 Redis ──> WS-6 Postgres ──> WS-7 object storage ──> WS-8 Qdrant ──> WS-9 queue
WS-10 observability ──> WS-11 metrics ──> WS-12 logs ──> WS-13 tracing
WS-14 cost monitoring (runs throughout, tags from day one)
WS-15 disaster recovery (backs up everything above)
```

WS-1 is the compute foundation; WS-2 → WS-3 → WS-4 is the serving spine.
WS-5..WS-9 harden the data services that every plan above depends on.
WS-10..WS-13 are the observability stack (WS-10 first). WS-14 tags everything
from day one; WS-15 is the summit and the regular drill.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| GPU cluster | WS-1 |
| Model serving | WS-2 |
| Load balancing | WS-3 |
| Autoscaling | WS-4 |
| Redis caching | WS-5 |
| PostgreSQL | WS-6 |
| Object storage | WS-7 |
| Vector database | WS-8 |
| Queue system | WS-9 |
| Observability | WS-10 |
| Metrics | WS-11 |
| Logs | WS-12 |
| Tracing | WS-13 |
| Cost monitoring | WS-14 |
| Disaster recovery | WS-15 |

## Tests

```bash
make cluster-plan && make cluster-up          # WS-1
make deploy-serving                            # WS-2
make deploy-ingress                            # WS-3
make deploy-autoscale                          # WS-4
make deploy-redis                              # WS-5
make migrate-postgres && make backup-postgres  # WS-6
make upload-artifact / make pull-artifact      # WS-7
make snapshot-qdrant / make restore-qdrant     # WS-8
make deploy-queue                              # WS-9
make deploy-observability                      # WS-10..13
make deploy-cost                               # WS-14
make dr-drill                                  # WS-15 (full restore + eval gate)
```

Never commit secrets, backups, or cluster state; `infra/` generated state is
git-ignored except code. Infrastructure hosts every other plan (runtime
serving, RAG stores, agent sandbox, eval regression); the docker-compose dev
stack stays the CI/local mirror.
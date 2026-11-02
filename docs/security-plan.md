# Security — Implementation Plan

Goal: a cohesive security program for the entire KothaGPT stack — from model
and data threats (prompt injection, dataset poisoning, abuse) to platform
threats (tool authorization, sandboxing, secrets, auth, encryption, audit,
supply chain, artifact integrity) — culminating in a red-team program. Security
is a release gate across every other plan, not a feature.

Guiding principles:

- Build on what exists: several security controls are already scoped elsewhere —
  agent sandboxing (`docs/agent-plan.md` WS-15), API auth/rate limiting
  (`docs/runtime-plan.md` WS-12/WS-13), PII filtering (`docs/dataset-pipeline-plan.md`
  B4), audit logs (enterprise/agents), encryption (`docs/infra-plan.md`). This
  plan makes them a *cohesive, tested, owned* program and fills the gaps
  (prompt injection, tool authorization, secret isolation, abuse detection,
  poisoning, supply chain, red-team).
- Defense in depth: no single control is trusted; the model, the tool layer,
  and the platform each enforce their own boundary.
- Everything untrusted is an input: prompts, tool results, retrieved chunks,
  uploaded documents, and third-party packages all get validation before
  influence.
- Security is testable: each control has a negative test (attack) and a
  positive test (legit use still works); the red-team program is scheduled, not
  sporadic.
- Least privilege everywhere: roles, keys, buckets, sandboxes, and model
  permissions are all scoped to the minimum needed.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Agent sandboxing | scoped in `docs/agent-plan.md` WS-15 (OS isolation, no network, resource limits) | not built; needs shared policy with security owner |
| API auth / rate limiting | scoped in `docs/runtime-plan.md` WS-12/WS-13 | not built; no key scope model beyond runtime |
| PII | `data/pipeline/quality.py` PII detection + redaction (dataset plan B4) | no runtime PII guard on outputs |
| Encryption | `docs/infra-plan.md` (object storage SSE, TLS) | no key management policy, no at-rest inventory |
| Audit | enterprise tier + agent transcripts scoped | no unified audit log schema/retention |
| Prompt injection / tool authorization / secrets / abuse / poisoning / supply chain / artifacts / red-team | — | whole workstreams missing |

---

## Workstreams

### WS-1 — Prompt injection protection (`services/security/injection.py`)

Goal: untrusted content cannot hijack the model's instructions.

- Layer defenses: input classification (instruction-delimiter / jailbreak /
  hidden-instruction detection), robust chat templating that quotes retrieved
  + tool content as data (RAG/agent context is never concatenated as
  instructions), and a policy check on final output (does it leak tools/roles?).
- Cover the attack surfaces: direct chat, RAG context, tool results, file
  uploads, agent messages.
- Deliverables: injection classifier + template hardening, attack fixture set.
- Metric: attack fixture suite (direct, RAG-injected, tool-result-injected)
  blocked ≥ threshold; benign prompts unaffected (false-positive rate ≤
  threshold).

### WS-2 — Tool authorization (`services/security/authz.py`)

Goal: tool calls execute only with explicit, scoped authorization.

- Central authorization check at tool dispatch (replacing per-tool ad-hoc
  checks): resource, action, budget, and sandbox profile from the permission
  matrix (`docs/agent-plan.md` WS-14); high-risk tools (code, db, browser)
  require per-call or per-session approval.
- Enforced at the boundary (the API/tool layer), not inside the model.
- Deliverables: `authz.py`, approval flows, denial audit.
- Metric: every dispatch is authorized (negative test: unauthorized call
  never executes); approval flows round-trip; all denials are auditable.

### WS-3 — Agent sandboxing (`services/security/sandbox.py`)

Goal: defense-in-depth isolation (productionizing agent-plan WS-15).

- OS/container isolation per run with network egress allow-list, scratch-only
  filesystem, CPU/mem/time budgets, and guaranteed teardown; a shared
  `sandbox_policy` consumed by every tool.
- Deliverables: hardened sandbox + escape test suite.
- Metric: escape tests (network, file, process, container) all blocked; no
  sandbox survives past its timeout; resource limits enforced under load.

### WS-4 — Secret isolation (`services/security/secrets.py`)

Goal: no credential in code, config, logs, or memory dumps.

- Central secrets manager (vault-style) with per-service roles, rotation, and
  versioning; env/config files hold references only; emission-time redaction of
  any secret-looking value from logs and traces (runtime WS-12).
- Deliverables: secrets client, rotation policy, leak scanner (CI).
- Metric: zero secrets in git history (scan passes); rotation revokes within
  documented time; redaction verified on secret-shaped fixtures.

### WS-5 — API authentication (`services/security/auth.py`)

Goal: every `/v1/*` request is authenticated (runtime WS-12 productionized).

- Bearer API keys (hashed at rest), optional JWT/SSO, per-key scopes and model
  ACLs, revocation + rotation, one-time display; enforced by a central
  dependency so no router can bypass.
- Deliverables: auth dependency, key admin, negative tests.
- Metric: no 200 without a valid key; revoked keys rejected immediately; scopes
  enforced per endpoint; keys stored hashed.

### WS-6 — API rate limiting (`services/security/ratelimit.py`)

Goal: protect the runtime from burst and abuse (runtime WS-13 productionized).

- Token-aware limits (requests + tokens/min) per key/tenant via Redis,
  sliding-window with burst, per-model quotas, `429` + `Retry-After`, aligned
  to per-device throughput budgets.
- Deliverables: ratelimit middleware, config, load test.
- Metric: burst throttled at ceiling; legit traffic unaffected; 429s
  well-formed and counted; bypass attempts fail.

### WS-7 — Data encryption (`services/security/crypto.py`)

Goal: data is encrypted in transit and at rest, keys managed.

- TLS everywhere (infra plan); at-rest encryption inventory (object storage,
  Postgres, Qdrant, Redis, model artifacts) with a single KMS policy and
  per-service key hierarchy.
- Deliverables: encryption inventory + KMS wiring, key lifecycle docs.
- Metric: every storage class is encrypted at rest; key rotation works without
  data loss; inventory is CI-checked.

### WS-8 — Runtime PII detection (`services/security/pii_guard.py`)

Goal: PII never leaves the system in outputs.

- Reuse `data/pipeline/quality.py` detectors as a runtime output guard (mask or
  reject PII in responses, chat logs, tool results); per-tenant policy
  (mask vs drop vs alert).
- Deliverables: output PII guard + fixtures.
- Metric: PII output blocked/masked at runtime; benign text unaffected;
  policy configurable per tenant.

### WS-9 — Audit logs (`services/security/audit.py`)

Goal: every security-relevant action is recorded, queryable, retained.

- Unified audit schema (actor, action, resource, decision, trace id, tenant,
  timestamp); wired into auth, tool dispatch, memory writes, KB changes, model
  publishes, admin actions; tamper-evident (append-only + hash chain) and
  retained per policy.
- Deliverables: audit client, retention, query API.
- Metric: every listed action produces an audit record; chain is verifiable;
  retention honored; audit queries return full context.

### WS-10 — Model abuse detection (`services/security/abuse.py`)

Goal: detect and limit misuse (spam, automated abuse, policy violations).

- Usage analytics on top of audit + metrics: detect abusive patterns (bursts,
  policy-evasion attempts, abuse of free tier) and trigger the rate-limit /
  suspend flows; a review queue for flagged tenants.
- Deliverables: abuse detector + review UI.
- Metric: synthetic abuse bursts are flagged/limited; legit tenants unaffected;
  review decisions audited.

### WS-11 — Dataset poisoning detection (`data/pipeline/security.py`)

Goal: poisoned or backdoored data never trains or ships.

- Extend the dataset pipeline with poisoning checks: outlier/backdoor heuristics
  (label-flip signatures, trigger-phrase clustering, source trust scoring), a
  provenance chain from raw source to shard (already content-addressed), and
  optional differential analysis against held-out clean sets.
- Deliverables: `data/pipeline/security.py` + poison fixture tests.
- Metric: injected poison fixtures are flagged before release; legitimate data
  passes; provenance chain is complete per record.

### WS-12 — Supply-chain security (`scripts/sbom.py` + CI gate)

Goal: dependencies are known, scanned, and pinned.

- Generate SBOMs (Python, Node, Rust, Go) in CI, scan for known CVEs + license
  violations, pin lockfiles everywhere, verify package provenance (pinned
  digests / signed releases where available), and alert on new-vulnerability
  diffs per release.
- Deliverables: SBOM generator, scanning gate, pinning policy.
- Metric: CI fails on a known-CVE diff; SBOM regenerated per release; lockfiles
  are the source of truth.

### WS-13 — Secure model artifacts (`ml/security/artifacts.py`)

Goal: weights and artifacts are authentic and tamper-evident.

- Sign + digest every model artifact (checkpoint, quantized weights, tokenizer,
  config) at build; verify at load (loader in `docs/runtime-plan.md` WS-8);
  artifacts distributed only from the registry, never from URLs.
- Deliverables: signing/verification, artifact policy.
- Metric: a tampered artifact fails to load; signature chain verifies for every
  release artifact; no unsigned artifact is servable.

### WS-14 — Red-team testing (`evals/suites/security.yaml` + `scripts/redteam.py`)

Goal: a scheduled adversarial program over all the above.

- Maintain an adversarial suite (injection, jailbreak, PII leak, tool abuse,
  sandbox escape, auth bypass, poison) run on every release like the eval
  regression gate (`docs/eval-plan.md` WS-14); plus periodic human-led red-team
  sessions with a findings tracker and owner.
- Deliverables: security eval suite, red-team runbook, findings tracker.
- Metric: suite passes every release; red-team findings have owners and close
  within a documented SLA; zero criticals open at release.

---

## Sequencing & dependencies

```
WS-1 injection ──> WS-2 tool authorization ──> WS-3 sandboxing (agent runtime)
WS-5 auth ──> WS-6 rate limiting ──> WS-10 abuse detection
WS-4 secrets ──> WS-7 encryption (infra) ──> WS-9 audit
WS-8 PII guard (dataset B4 reuse) ──> WS-11 poisoning (dataset pipeline)
WS-12 supply chain ──> WS-13 artifact signing (runtime loader)
WS-14 red team (runs over everything, gates releases)
```

WS-1..WS-3 protect the model/tool layer; WS-4..WS-7 + WS-9 protect the
platform; WS-8/WS-11 protect data in and out; WS-12/WS-13 protect the
pipeline; WS-14 is the summit and the release gate.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Prompt injection protection | WS-1 |
| Tool authorization | WS-2 |
| Agent sandboxing | WS-3 |
| Secret isolation | WS-4 |
| API authentication | WS-5 |
| API rate limiting | WS-6 |
| Data encryption | WS-7 |
| PII detection | WS-8 |
| Audit logs | WS-9 |
| Model abuse detection | WS-10 |
| Dataset poisoning detection | WS-11 |
| Supply-chain security | WS-12 |
| Secure model artifacts | WS-13 |
| Red-team testing | WS-14 |

## Tests

```bash
python -m pytest tests/test_injection.py               # WS-1
python -m pytest tests/test_tool_authz.py              # WS-2
python -m pytest tests/test_sandbox_escapes.py         # WS-3
make secrets-scan                                       # WS-4
python -m pytest tests/test_auth.py tests/test_rate_limit.py  # WS-5/6
make encrypt-inventory                                  # WS-7
python -m pytest tests/test_pii_guard.py               # WS-8
python -m pytest tests/test_audit.py                   # WS-9
python -m pytest tests/test_abuse_detection.py         # WS-10
python -m pytest tests/test_poison_detection.py        # WS-11
make sbom-check                                         # WS-12
python -m pytest tests/test_artifact_signing.py        # WS-13
make eval-security && make redteam-drill               # WS-14
```

Never commit secrets, keys, or signed artifacts' private material; `services/security/`
state is git-ignored. This plan is the hard gate referenced by every other
plan; a security failure blocks a release even when all functional evals pass.
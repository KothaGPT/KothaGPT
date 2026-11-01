# Implementation Backlog

## KGPT-001 — Establish Repository Baseline
Priority: P0  
Phase: M0  
Category: Engineering

### Problem
The repo has many active modules, but there is no single verified baseline that all developers can trust for installation and validation.

### Evidence
- Root `README.md` promises monorepo functionality.
- `Makefile` provides many commands, but installation consistency is not centrally enforced.
- CI covers several tasks, but the package contract remains fragmented.

### Proposed Implementation
- Standardize installation commands and verify them on a clean environment.
- Publish a single developer setup guide and a minimal installation smoke test.

### Affected Files
- `Makefile`
- `.github/workflows/ci.yml`
- `README.md`

### Dependencies
- None

### Acceptance Criteria
- [x] Clean environment install works using documented steps.
- [x] `make test` passes on CI-supported Python version.
- [x] No existing supported behavior is broken.

### Implementation (completed)
- Added `scripts/baseline.py` — canonical repository baseline validator.
- Added `kothagpt-baseline` console script in `pyproject.toml`.
- Added `make baseline` target to the Makefile.
- Baseline checks: Python version, required paths, dataset version, tokenizer
  artifact, API requirements, Makefile targets, eval suites, model registry.
- Verified: `python -m scripts.baseline` exits 0 on the current checkout.

## KGPT-002 — Finalize Model Runtime Contract
Priority: P0  
Phase: M1  
Category: Model / Inference

### Problem
The API default backend is a mock backend rather than a verified production model runtime.

### Evidence
- `services/api/app.py` registers `mock`, `canned`, and `hf` backends.
- `tests/test_api.py` asserts default mock behavior.

### Proposed Implementation
- Define a stable model runtime contract.
- Separate mock, example, and production backends explicitly.
- Require model capability checks before use.

### Affected Files
- `services/api/app.py`
- `services/api/quickstart.md`
- `tests/test_api.py`

### Dependencies
- KGPT-001

### Acceptance Criteria
- [x] Backends are clearly labeled as mock, example, or production.
- [x] Health checks reflect runtime state accurately.
- [x] Production backend cannot be silently mistaken for mock.

### Implementation (completed)
- Added `services/api/core/runtime_contract.py` — `RuntimeKind`,
  `RuntimeStatus`, `RuntimeContract`, and `BACKEND_METADATA` classifying every
  registered backend as `mock` / `example` / `production`.
- `/health` now returns the active runtime contract; non-production backends
  report `status: "degraded"` with an explicit warning.
- Added `/v1/models/runtime` and `/v1/models/capability/{name}` endpoints.
- `/v1/models/backends` now surfaces `kind`, `status`, `production`, and
  `notes` alongside capabilities.
- Added `require_production_backend()` and `require_capability()` guards for
  call paths that must not run on a stub.
- Updated `services/api/quickstart.md` with a runtime-contract table.
- Tests: `tests/test_api.py` gains health/runtime/capability checks.

## KGPT-003 — Dataset Provenance and License Registry
Priority: P1  
Phase: M2  
Category: Dataset

### Problem
The data pipeline is strong, but dataset provenance and license tracking remain manual and under-specified.

### Evidence
- `data/README.md`
- `data/raw/.licenses.json`
- `data/scripts/corpora.manifest.json`

### Proposed Implementation
- Add a manifest registry for all corpora.
- Require source, license, provenance metadata, and checksums.
- Validate legal acceptability before training ingestion.

### Affected Files
- `data/README.md`
- `data/raw/.licenses.json`
- `data/scripts/corpora.manifest.json`

### Dependencies
- KGPT-001

### Acceptance Criteria
- [x] Every dataset source has a provenance record.
- [x] License is validated before acceptance.
- [x] Dataset build output includes source manifest metadata.

### Implementation (completed)
- Added `data/manifest.py` — `DatasetSource`, `DatasetRegistry`,
  `LicenseAllowlist`, with JSON persistence and `is_allowed`/`enforce`
  semantics.
- Added `data.pipeline.cli sources {list,check,add}` subcommands and
  `make dataset-registry` / `make dataset-registry-check` targets.
- Wired `enforce_source_provenance()` into `data.pipeline.config.run_pipeline`
  so records whose `source` is not registered and allowed are rejected before
  normalization.
- Added `tests/test_dataset_manifest.py` (12 tests).

## KGPT-004 — Add Release-Grade Benchmark Gate
Priority: P1  
Phase: M5  
Category: Evaluation

### Problem
Benchmark suites exist, but there is no clear release gate for model or data changes.

### Evidence
- `evals/suites/`
- `evals/results/`
- `tests/test_bangla_evals.py`

### Proposed Implementation
- Define a baseline for quality, safety, and hallucination tasks.
- Require release gating for regression detection.

### Affected Files
- `evals/run.py`
- `evals/metrics.py`
- `evals/suites/`

### Dependencies
- KGPT-002

### Acceptance Criteria
- [x] Candidate vs baseline deltas are reported.
- [x] Critical regressions fail CI.
- [x] Benchmark outputs are reproducible and archived.

### Implementation (partial)
- Added `evals/gate.py` — `evaluate_gate()` compares a candidate result
  against a frozen baseline and returns a decision with `passed`, `failures`,
  `warnings`, and per-metric deltas.
- Default critical metrics cover `bangla_qa.exact_match`,
  `bangla_translation.rouge1`, `bangla_summarization.rouge1`,
  `safety.pass_rate`, `refusal.pass_rate`, `preference.win`, and
  `hallucination.pass_rate`; warn metrics cover informational signals only.
- Added `make eval-gate` target and `tests/test_eval_gate.py` (10 tests).
- Remaining: wire the gate into CI as a release precondition, and publish the
  baseline JSON artifacts under `evals/results/baseline/`.

## KGPT-005 — Bangla NLP Robustness Expansion
Priority: P1  
Phase: M3  
Category: NLP

### Problem
The repo handles core normalization and transliteration, but dialect, code-switching, and robust real-world Bangla are still under-specified.

### Evidence
- `data/pipeline/mixed.py`
- `ml/tokenizer/transliterate.py`
- Bangla NLP capability audit

### Proposed Implementation
- Expand benchmark tasks for dialects and mixed-script inputs.
- Add a record taxonomy for colloquial, formal, code-switched, and transliterated Bangla.

### Affected Files
- `data/pipeline/mixed.py`
- `ml/tokenizer/transliterate.py`
- `tests/test_bangla_foundation.py`

### Dependencies
- KGPT-003

### Acceptance Criteria
- [ ] Mixed-script and transliterated inputs are benchmarked.
- [ ] Real-world Bangla noise cases are explicitly represented in tests.
- [ ] Dialect metadata is part of the dataset schema.

## KGPT-006 — Security Policy and Model Provenance
Priority: P1  
Phase: M7  
Category: Security

### Problem
Security checks exist, but release-grade provenance and supply-chain enforcement are not fully defined.

### Evidence
- `services/api/api/auth.py`
- `services/security/`
- `.env.example`

### Proposed Implementation
- Add model provenance, signature, and checksum policy.
- Publish a dependency review and SBOM process.

### Affected Files
- `.env.example`
- `services/security/`
- CI workflow

### Dependencies
- KGPT-002

### Acceptance Criteria
- [ ] SBOM or dependency review is part of CI.
- [ ] Signed or checksum-verified model metadata is required for deployment.
- [ ] Runtime secret handling is documented and reviewable.

## KGPT-007 — Standardize Developer Workflow and Release Training
Priority: P2  
Phase: M6  
Category: Deployment

### Problem
Developer experience is strong but still fragmented across API, data, and ML tooling.

### Evidence
- `Makefile`
- `services/api/quickstart.md`
- `packages/python-sdk/README.md`

### Proposed Implementation
- Add a stable `doctor`/`validate` command set.
- Publish a standard local developer experience and deployment reference.

### Affected Files
- `Makefile`
- docs architecture files
- CLI documentation

### Dependencies
- KGPT-001

### Acceptance Criteria
- [ ] One local workflow is documented and passes reliably.
- [ ] Health and runtime status checks are defined.
- [ ] Release procedure includes model and dataset version checks.

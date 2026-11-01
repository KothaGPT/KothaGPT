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
- [ ] Clean environment install works using documented steps.
- [ ] `make test` passes on CI-supported Python version.
- [ ] No existing supported behavior is broken.

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
- [ ] Backends are clearly labeled as mock, example, or production.
- [ ] Health checks reflect runtime state accurately.
- [ ] Production backend cannot be silently mistaken for mock.

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
- [ ] Every dataset source has a provenance record.
- [ ] License is validated before acceptance.
- [ ] Dataset build output includes source manifest metadata.

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
- [ ] Candidate vs baseline deltas are reported.
- [ ] Critical regressions fail CI.
- [ ] Benchmark outputs are reproducible and archived.

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

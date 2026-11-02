# Developer Experience and Deployment Audit

## Current state

The repository contains a credible developer workflow for local engineering and experimentation:

- `Makefile` provides local commands for data build, tokenizer build, smoke training, and evaluation.
- `services/api/quickstart.md` explains how to run a prototype API.
- `packages/python-sdk/README.md` documents how to call the API from Python.
- `.env.example` defines a wide runtime environment contract.

## What is already good

- Local install flow is documented.
- API backend selection is explicit (`mock`, `canned`, `hf`).
- `make dev`, `make data`, `make tokenizer`, and `make test` provide a workable local cycle.
- CI checks are defined in `.github/workflows/ci.yml`.

## Gaps in developer experience

### 1. Installation path is fragmented

The project has multiple runtime sub-environments (`services/api`, `ml`, `npm`, `Cargo`, `Go`), but there is no unified installation guarantee across all of them.

### 2. Health and readiness are not yet standardized

The FastAPI app exposes `/health`, but there is no production readiness contract that verifies model availability, dataset version, tokenizer version, and runtime health together.

### 3. CLI maturity is still aspirational

The project sketches a CLI (`kothagpt`), but the evidence shows a prototype rather than a complete operating contract.

### 4. Deployment story is thin

Docker and infrastructure directories exist, but there is no clearly documented production deployment path backed by a validated configuration contract.

## Recommended developer workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r services/api/requirements.txt
make data-validate
make tokenizer-build
make test
make dev
```

Recommended commands for the next stage:

```bash
kothagpt doctor
kothagpt data validate
kothagpt data prepare
kothagpt model info
kothagpt train
kothagpt evaluate
kothagpt chat
kothagpt serve
```

These commands are not yet evidence-backed as fully implemented; they should be introduced only after the underlying capabilities are verified.

## Deployment recommendations

1. Keep local development and CI lightweight.
2. Use Docker for the API service only after a validated runtime is in place.
3. Publish model and dataset version metadata in every release.
4. Separate development, staging, and production config as distinct environments.
5. Add health, readiness, and metrics endpoints before public deployment.

## Definition of done

A user should be able to:

- install the project with one supported command set
- run local validation checks
- start the API with an explicit backend mode
- validate tokenizer and dataset versions
- evaluate a model using a benchmark suite
- understand which features are supported versus experimental

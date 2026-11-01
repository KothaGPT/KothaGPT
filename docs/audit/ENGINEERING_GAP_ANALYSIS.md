# Engineering Gap Analysis

## Summary

KothaGPT contains substantial implementation work in data preprocessing, tokenizer experimentation, evaluation, and API scaffolding. However, multiple engineering issues still block a clean progression to production. The biggest problem is not a total lack of work; it is the gap between documented ambition and actual end-to-end integration, packaging discipline, and verification.

## Gap inventory

| Gap ID | Component / File | Current behavior | Evidence | Expected behavior | Impact | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| KG-001 | Repository packaging and installability | The repo exposes a monorepo but lacks a single root Python package and a unified install contract. | Root `README.md` describes a platform, but `pyproject.toml` is absent at the root. Python modules rely heavily on `import` paths such as `from data.pipeline ...` and `from services.api.app ...` rather than a single package entry. | A clear root package contract or explicit workspace layout with validated installation steps. | Developer onboarding is inconsistent; CI and local setups can diverge. | P1 |
| KG-002 | Model training / runtime production integration | The repo contains model definitions and training commands, but the actual runtime path for serving a real model is not connected to a stable production pipeline. | `ml/models/layers.py`, `ml/models/config.py`, and `Makefile` define training loops and model config, while `services/api/quickstart.md` still describes a mock or canned default backend. | Default production runtime should either use a verified model backend or explicitly document the mock-only status. | Users cannot trust that the API is model-backed by default. | P0 |
| KG-003 | API/backend maturity | The default backend is `mock`; HF backend is optional and degrades to canned output when dependencies are missing. | `services/api/app.py` registers `mock`, `canned`, and `hf`; `tests/test_api.py` asserts mock behavior by default. | Production-ready deployment should document backend selection, failure modes, and required model validation before using a real inference backend. | Product demos may look production-like while remaining non-production. | P0 |
| KG-004 | Dataset provenance and release discipline | The project encourages external corpora under `data/raw/` and uses license metadata patterns, but no central dataset registry or provenance ledger is enforced by code. | `data/README.md`, `data/scripts/corpora.manifest.json`, and `data/raw/.licenses.json` show manual human review is required. | A formal dataset manifest and registry with source, license, provenance, and checksum tracking. | Dataset quality and legal compliance remain brittle. | P1 |
| KG-005 | Model artifact and checkpoint governance | The repository uses `ml/tokenizer/artifacts`, `ml/pretrain/artifacts`, and `ml/sft/artifacts` but these are not wired into a canonical model registry. | `Makefile` references output directories; `ml/README.md` says never commit model weights or private datasets. | Versioned model registry with metadata, config digest, and release notes. | Reproducibility and experiment comparison are weak. | P1 |
| KG-006 | Benchmarking and release gates | Evaluation exists, but benchmark results are not clearly structured as release gates or acceptance criteria for model changes. | `evals/results/` contains dated JSON and Markdown reports; suites exist under `evals/suites/`. | Every release should compare baseline vs candidate across Bangla quality, safety, hallucination, and SFT tasks. | Quality regressions can go unnoticed. | P1 |
| KG-007 | Security verification and supply chain controls are present but incomplete | Security checks and tests exist, but the repo does not demonstrate a formal dependency and model provenance policy. | Security-related modules and tests exist (`services/security/`, `tests/test_security_injection.py`, `tests/test_sandbox_escapes.py`), but there is no clear SBOM / supply chain publication path. | A documented security and dependency policy with SBOM generation and vendor review. | Production risk is moderated but not eliminated. | P1 |
| KG-008 | Documentation drift | The project documentation is rich and ambitious, but some plan documents are more aspirational than actual code state. | `README.md`, `docs/foundation.md`, and `TODO` describe a broad product vision, while many modules remain scaffolded. | The docs should clearly distinguish implemented, partial, planned, and unverified capabilities on every major component. | Team velocity and user trust are reduced by ambiguity. | P2 |
| KG-009 | Configuration management and environment validation | `.env.example` outlines many runtime variables, but there is no centralized config validation layer or schema enforcement across services. | `configs/model.yaml` exists, but the root runtime config pattern is only partially enforced. | A validated config contract for model, training, API, storage, and security settings. | Configuration errors can be difficult to diagnose in production. | P2 |
| KG-010 | Multi-language platform scaffolding is broader than validated core | The repo is structured for Python, Rust, Go, TypeScript, and web app support, but the validated backend is still narrow. | `README.md` advertises SDKs and platform layers; language-specific packages exist, but evidence for complete integration remains limited. | Limit scope to the core stack that is proven and gradually add adapters after validation. | Platform breadth increases maintenance cost without verified return. | P2 |

## Root causes

1. The project is pursuing a broad full-stack research platform while the underlying model/data/runtime stack is still maturing.
2. Documentation and implementation status are not always aligned; there are “done” plan references mixed with placeholder code.
3. Strong component-level work exists, but the project still lacks a single canonical production baseline.
4. The repo is designed for experimentation, but not all release-grade boundaries and operational controls are enforced.

## Recommended solution

1. Establish a single verified baseline for install, tests, data, and tokenizer output.
2. Define a stable model runtime contract and a clear default backend.
3. Publish a dataset registry and model registry with metadata, checksums, and provenance.
4. Convert evaluation plans into release gate requirements.
5. Separate supported, experimental, and aspirational features in documentation and tooling.

## Acceptable outcomes

- One root-level installation path works on clean machines.
- The API defaults are intentional and documented.
- Data and model releases are versioned and reproducible.
- Benchmark results are compared against fixed baselines.
- Security and dependency controls are visible, enforceable, and reviewed.

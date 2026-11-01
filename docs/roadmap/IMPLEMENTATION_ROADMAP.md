# Implementation Roadmap

## Overview

This roadmap converts the repository evidence into a staged plan that keeps the project moving without forcing an architectural rewrite. The roadmap is anchored to the actual repository state: data pipeline and tokenizer work are relatively strong, API scaffolding is active, evaluation infrastructure exists, and model runtime maturity remains the largest gap.

## M0 — Repository Stabilization
Objective: establish a verified baseline.
Current problem: there is no single clean baseline that all developers can trust.
Implementation tasks:
- verify installation path
- standardize local developer commands
- lock a supported Python version in docs and CI
- ensure test and lint coverage is stable
Dependencies: none
Deliverables: verified developer workflow, stable CI, test smoke checks
Acceptance criteria: clean environment install works; tests pass; docs align with actual repo state
Priority: P0

## M1 — Architecture and Interface Foundation
Objective: create stable boundaries between API, model, data, and evaluation.
Current problem: wide scope and documented ambition exceed canonical runtime boundaries.
Implementation tasks:
- define model runtime contract
- define dataset and model registry boundaries
- standardize config validation
- map supported vs experimental features
Dependencies: M0
Deliverables: runtime contract, config schema, architecture docs, supported feature matrix
Acceptance criteria: backends are explicit; config validation exists; supported features are clearly tagged
Priority: P1

## M2 — Dataset Engineering
Objective: make the data pipeline reproducible, reviewed, and legal.
Current problem: pipeline exists but provenance and licensing discipline are manual.
Implementation tasks:
- define manifest and license registry
- build quality gates and leakage checks
- formalize splits and data cards
- publish reproducible dataset versioning
Dependencies: M0, M1
Deliverables: dataset manifest, schema, versioned builds, release report
Acceptance criteria: no untracked source can be used for training; all builds have provenance
Priority: P1

## M3 — Bangla NLP Improvements
Objective: strengthen real-world Bangla quality and robustness.
Current problem: normalization and transliteration are implemented, but dialect and code-switching support are still thin.
Implementation tasks:
- expand mixed-script and transliteration benchmarks
- add dialect metadata and coverage checks
- formalize informal and code-switched evaluation
Dependencies: M2
Deliverables: new benchmark tasks, dataset taxonomy, updated evaluation docs
Acceptance criteria: mixed-script real-world cases are in the benchmark suite and pass quality thresholds
Priority: P1

## M4 — Model and Training Pipeline
Objective: prepare a realistic fine-tuning and experimentation pipeline.
Current problem: model design and training commands exist, but production runtime is not yet proven.
Implementation tasks:
- define base model strategy and release path
- define LoRA/QLoRA or SFT path with clear metrics
- standardize model registry and checkpoint metadata
Dependencies: M1, M2, M3
Deliverables: model registry, training docs, checkpoint release rules
Acceptance criteria: training experiments are reproducible and trackable
Priority: P1

## M5 — Evaluation and Benchmarking
Objective: make quality visible and measurable.
Current problem: benchmark suites exist but must be enforced as standard release gates.
Implementation tasks:
- formalize baseline and regression workflow
- add benchmark results to CI and release review
- separate safety, language quality, and hallucination metrics
Dependencies: M4
Deliverables: benchmark spec, result reports, regression gate
Acceptance criteria: quality regressions fail release review
Priority: P1

## M6 — Developer Experience and Deployment
Objective: make the project usable by developers and researchers.
Current problem: the repo is broad but not yet frictionless to install or operate.
Implementation tasks:
- finalize CLI and config workflow
- define health endpoints and deployment metadata
- document one supported deploy path
Dependencies: M0, M1, M5
Deliverables: deployment guide, health contract, CLI workflow
Acceptance criteria: a developer can run the project end-to-end without undocumented steps
Priority: P2

## M7 — Security and Responsible AI
Objective: prepare the project for broader public or enterprise use.
Current problem: security features exist but need governance and release-grade controls.
Implementation tasks:
- add model provenance and dependency review
- continue safety and refusal testing
- define privacy and retention policy for user data
- publish model cards and limitations
Dependencies: M2, M4, M5
Deliverables: security playbook, risk register, model card
Acceptance criteria: model and data deployments are documented, reviewed, and safe-by-default
Priority: P1

## Dependency-aware delivery order

1. M0 — Repository Stabilization
2. M1 — Architecture and Interface Foundation
3. M2 — Dataset Engineering
4. M3 — Bangla NLP Improvements
5. M4 — Model and Training Pipeline
6. M5 — Evaluation and Benchmarking
7. M6 — Developer Experience and Deployment
8. M7 — Security and Responsible AI

## Definition of Done

The project is ready for the next significant engineering phase when:

- installation is stable and documented
- model backend default is explicit
- data provenance and versioning are enforceable
- benchmark release gates exist
- security policy and model limitations are public

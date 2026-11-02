# Target Architecture

## Goal

Define a maintainable and realistic architecture for the next engineering phase of KothaGPT without replacing the repository’s main direction.

## Target structure

```text
KothaGPT/
├── src/
│   └── kothagpt/
│       ├── core/
│       ├── data/
│       ├── nlp/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── api/
│       ├── cli/
│       └── shared/
├── configs/
├── datasets/
├── experiments/
├── models/
├── benchmarks/
├── scripts/
├── tests/
├── docs/
├── examples/
├── pyproject.toml
├── README.md
└── Makefile
```

## Current mapping to the repo

- `data/` is the de facto dataset layer and should stay as a separate concern.
- `ml/` should become the canonical model and training boundary; today it is already close to that role.
- `services/api` should remain the API boundary, but its default mock-only mode should be clearly separated from production model backends.
- `evals/` should become the canonical benchmark and release gate boundary.
- `packages/` and `apps/` should remain user-facing interfaces, not the core model and data layer.

## Module responsibilities

### Core
- configuration loading
- logging and tracing
- environment validation
- feature flags

### Data
- dataset manifests
- source licensing and provenance
- normalization and quality filters
- split and versioning
- benchmarks and eval fixtures

### NLP
- Bangla pipeline normalization
- script detection
- transliteration support
- tokenization and vocabulary metadata

### Models
- model configuration and digesting
- tokenizer and embedding contracts
- training config validation
- inference adapter interface

### Training
- data loading and preprocessing
- reproducible training loops
- checkpointing and resume support
- experiment tracking

### Evaluation
- benchmark suite definitions
- metric aggregation
- regression checks and release gate thresholds

### Inference
- batching and request validation
- model loading and runtime selection
- prompt formatting and generation adapters
- latency and observability hooks

### API / CLI
- versioned API contracts
- auth and rate limiting
- CLI commands for doctor, validate, train, and evaluate

## Recommended boundaries

- Data layer must not depend on API or web code.
- Model code should depend on data contracts and configuration, not on the API layer.
- API and SDKs should depend on model and inference abstractions, not the reverse.
- Evaluation should be independent from training execution and should operate on versioned artifacts.

## Configuration contracts

The repo already has a useful pattern via `configs/model.yaml`, `.env.example`, and `ml/models/config.py`. The target state is a single validated configuration contract with:

- config schemas
- environment validation
- model digesting
- dataset version references
- training/evaluation metadata recording

## Model abstraction

The project should keep a general model abstraction for:

- generation
- embeddings
- reranking
- tool calling
- reward or preference scoring

The current decoder implementation in `ml/models/layers.py` is a good foundation, but real runtime selection and versioned deployment still need a stronger abstraction.

## Dataset abstraction

Use explicit data contracts for:

- source metadata
- language and script labels
- quality score and review state
- license and provenance
- split assignment and version ID

This aligns with the direction already visible in `data/pipeline` and the proposed schema patterns in the research docs.

## Inference and observability

- define health endpoints
- expose metrics for latency and tokens consumed
- log request IDs, model version, and model digest
- capture generation metadata separately from user prompt data

## Risk

The main risk is overengineering the platform before the core training-to-inference loop is verified. The target architecture should remain lean and modular enough to support current work without forcing a rewrite.

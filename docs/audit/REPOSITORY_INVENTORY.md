# Repository Inventory

## Scope and method

This inventory is based on direct inspection of the repository, including the root manifests, the `data/` pipeline, the `ml/` stack, the `services/` API, the `evals/` suite, and the test suite under `tests/`.

## 1. Project overview

KothaGPT is a monorepo for a Bangla-first conversational AI platform. The project aims to cover the full stack: corpus ingestion and cleaning, custom tokenization, model training, instruction tuning, evaluation, inference, RAG, agent tooling, and SDK/web exposure. The repository is best understood as an early-to-mid-stage engineering scaffold rather than a single production-ready product.

Evidence:
- Root `README.md` describes the stack as dataset → tokenizer → pretraining → instruction tuning → evaluation → inference → RAG → agents → SDKs → platform.
- `docs/foundation.md` describes the project as a Bangla-first, bilingual `bn+en` plan with tentative decisions still under review.
- `Makefile` includes data validation, tokenizer training, training smoke checks, and benchmark commands, indicating active development work rather than a static demo-only repo.

## 2. Architecture summary

The repository is organized into several functional layers:

- `data/`: raw corpus, pipeline logic, normalization, de-duplication, split logic, and versioned output.
- `ml/`: model configs, architecture, tokenizer experiments, training, inference, preference tuning, and SFT.
- `services/`: API layer, gateway, security, inference and RAG services.
- `evals/`: benchmark suites and metric code for Bangla evaluation.
- `apps/` and `packages/`: SDK and UI scaffolding.
- `configs/`: configuration files such as `configs/model.yaml`.
- `tests/`: dozens of focused unit tests for data, tokenizer, model config, API, security, and evaluation.

## 3. Technology stack

- Python: primary language for ML, dataset, evaluation, and service logic.
- PyTorch: model definition and training utilities.
- FastAPI: API service framework (`services/api/app.py`).
- pydantic + pydantic-settings: request models and configuration.
- PostgreSQL / Redis / Qdrant: declared in `README.md` and `.env.example` as intended runtime dependencies.
- TypeScript / Next.js: web app scaffolding under `apps/web`.
- Rust: `packages/rust-sdk` and runtime placeholders exist.
- Go: `packages/go-sdk` exists.

## 4. Core modules and inventory

### Data pipeline
- `data/README.md`: describes the raw → normalize → quality filter → deduplicate → split → version pipeline.
- `data/pipeline/cli.py`: CLI entry point for `run`, `normalize`, `check`, and `version` commands.
- `data/pipeline/normalize.py`: Unicode and HTML/text normalization logic.
- `data/pipeline/quality.py`: Bangla detection, PII detection, spam/toxicity filters.
- `data/pipeline/dedup.py`, `split.py`, `version.py`, `io.py`: deduplication, data splitting, versioning, and shard export.
- `data/raw/.licenses.json`: explicit license metadata pattern for a source corpus.

### Model and tokenizer
- `ml/models/config.py`: `ModelConfig`, `TrainingConfig`, and `DataConfig` dataclasses.
- `ml/models/layers.py`: decoder-transformer implementation with RMSNorm, RoPE, SwiGLU, and attention blocks.
- `ml/tokenizer/README.md`: custom Bangla tokenizer training with BPE, Unigram, and WordPiece implementations.
- `ml/tokenize_shards.py`: pre-tokenization utility for block shards.
- `ml/inference/`: batching, engine, KV cache, loader, registry, router; these exist as implementation scaffolding.

### Training and evaluation
- `Makefile`: commands for data validation, tokenizer build, smoke training, SFT, benchmarking, and safety evaluation.
- `evals/run.py`: evaluation runner for benchmark suites.
- `evals/metrics.py`: metrics such as exact match, ROUGE, language detection, hallucination coverage proxies.
- `evals/suites/`: multiple YAML suites for Bangla, safety, refusal, preference, and SFT families.
- `evals/results/`: report artifacts in JSON and Markdown.

### Services and API
- `services/api/app.py`: FastAPI app with mock, canned, and Hugging Face example backends.
- `services/api/quickstart.md`: developer quickstart for the prototype API.
- `services/api/api/auth.py`: bearer-token auth enforcement via env variables.
- `services/security/`: injection sanitization and security utilities.
- `services/gateway`, `services/inference`, `services/rag`, `services/agents`: service boundaries exist but are not yet unified into one production deployment model.

### SDKs and clients
- `packages/python-sdk/README.md`: Python client examples for chat, embeddings, rerank, tools, agents, and async usage.
- `packages/cli`: CLI package scaffolding.
- `packages/typescript-sdk`, `packages/go-sdk`, `packages/rust-sdk`: package placeholders with basic docs and manifests.

## 5. Existing capabilities

Verified implemented features include:

- Bangla text normalization utilities and punctuation handling.
- Mixed-script detection and spacing rules.
- Transliteration between Latin and Bangla script.
- Data pipeline validation for language, length, PII, spam, toxic content, deduplication, and splitting.
- A custom decoder-style model definition.
- A tokenizer framework with BPE / Unigram / WordPiece experiments.
- A FastAPI prototype with mock, canned, and Hugging Face example backends.
- Evaluation harnesses and benchmark suites for Bangla quality, hallucination, safety, refusal, and SFT.
- Security checks for injection and output leakage.

## 6. Development workflow

The repository uses a Makefile-driven workflow:

- `make bootstrap`
- `make dev`
- `make data`
- `make tokenizer`
- `make test`
- `make lint`
- `make eval-bangla`
- `make sft-train`

There is also a GitHub Actions pipeline in `.github/workflows/ci.yml` validating fixtures, tokenizer artifacts, evaluation suites, bootstrapping, and unit tests.

## 7. Known operational limitations

- There is no single root Python packaging file that binds all modules into one installable package.
- The training stack is documented more completely than it is operationally proven end-to-end.
- The default API backend is a mock backend, not a real model-backed runtime.
- Dataset corpus sources are expected to be downloaded externally under `data/raw/`; no curated private/public dataset is shipped in the repo.
- Large model weights and benchmarks are intentionally not committed, which is good for storage hygiene but makes reproducibility dependent on external artifacts.
- Several plan documents describe work as “done,” while the codebase still indicates a mostly scaffolded architecture; this makes project maturity harder to assess without a verified baseline.

## 8. Bottom line

KothaGPT is a strong engineering scaffold with real dataset, tokenizer, evaluation, and prototyping work already in place. It is not yet a fully integrated, production-grade Bangla AI platform. The repository most closely resembles an active research-and-early-product monorepo with a sound direction, but with implementation gaps still separating it from stable deployment.

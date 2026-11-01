# Kotha GPT — Bangla-First AI Platform

A monorepo scaffold for building a Bangla-first AI ecosystem:
dataset → tokenizer → pretraining → instruction tuning → evaluation → inference → RAG → agents → SDKs → platform.

## Stack

- Python: training, data, evaluation, orchestration
- Rust: high-performance runtime/components
- TypeScript: web platform and SDK
- PyTorch: model training
- FastAPI: API
- PostgreSQL: metadata
- Redis: cache/queues
- Qdrant/pgvector: vector search
- Next.js: web UI

## Repository layout

- `apps/` user-facing applications
- `services/` backend services (the HTTP API lives in `services/api`)
- `packages/` shared SDKs and libraries
- `ml/` model/training stack
- `data/` dataset pipeline
- `evals/` benchmarks and evaluation
- `infra/` deployment/infrastructure
- `docs/` engineering documentation
- `configs/` shared configuration
- `scripts/` developer automation

## Developer SDK

The platform exposes a REST + SSE + WebSocket API (`services/api`, `/v1/*`) with
first-party SDKs:

| SDK           | Location                  | Notes                                   |
| ------------- | ------------------------- | --------------------------------------- |
| Python        | `packages/python-sdk`     | `pip install kothagpt`                  |
| TypeScript    | `packages/typescript-sdk` | `@kothagpt/typescript-sdk`              |
| Go            | `packages/go-sdk`         | `kothagpt.dev/sdk/kothagpt`             |
| Rust          | `packages/rust-sdk`       | `kothagpt` crate                        |
| CLI           | `packages/cli`            | `kothagpt` executable                   |

Chat, streaming, tools, agents, embeddings, reranking, and model listing are
supported across every SDK. See `docs/sdk.md` for the full API reference and
`examples/` for runnable snippets.

## Quick start

```bash
cp .env.example .env
make bootstrap
make test
```

Bootstrap works on Linux, macOS, and Windows and previews its steps before
running them:

```bash
bash scripts/bootstrap.sh --dry-run                 # preview (auto-detected platform)
bash scripts/bootstrap.sh --dry-run --platform windows
make bootstrap                                      # execute without prompt
pwsh scripts/windows/bootstrap.ps1                  # native PowerShell alternative
```

Prerequisites: Node.js 18+ (corepack/pnpm). Missing Python, Go, and Rust
toolchains are installed automatically via the platform package manager
(apt/dnf/pacman, Homebrew, or winget); already-installed tools are never
touched. Optional: `scripts/install_rust` for a standalone Rust setup.

Run the API (mock backend) and try the CLI:

```bash
make serve-proto
kothagpt models
kothagpt chat --stream "বাংলায় হ্যালো"
```

Training is intentionally disabled by default. Configure a dataset and GPU environment before running training commands.


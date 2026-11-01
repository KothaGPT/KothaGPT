# Roadmap

Master checklist (auto-verified coverage): `docs/checklist.md`

## Phase 0 — Foundation (`docs/foundation-plan.md`)
- [x] AI-এর মূল উদ্দেশ্য নির্ধারণ (`docs/foundation.md` WS-1 Bangla-first platform)
- [x] Language scope (bn/en/multilingual) ঠিক করা (bn+en bilingual v1, `docs/foundation.md` WS-2)
- [x] Chat / Coding / Agent / Research AI নির্বাচন (chat-first, `docs/foundation.md` WS-3)
- [x] Target users নির্ধারণ (Bangla dev/student/enterprise, `docs/foundation.md` WS-4)
- [x] MVP feature list তৈরি (dashboard/playground/keys, `docs/web-app-plan.md` MVP)
- [x] Model size নির্ধারণ — 1B / 3B / 7B / 14B+ (small 768/12L `ml/configs/small.yaml`, sft 128/4L, long 16384)
- [x] License ও open-source strategy নির্ধারণ (MIT, `docs/foundation.md` WS-7)
- [x] Project name ও repository তৈরি (KothaGPT, `README.md`)
- [x] Repository bootstrap (`docs/bootstrap-migration-plan.md` — `scripts/bootstrap.sh` + `make bootstrap`)
- [x] CI (`.github/workflows/ci.yml` validate/plans/bootstrap/test + `actions/setup-python`)
- [x] coding standards (`make lint` ruff/cargo fmt/pnpm, `pyproject.toml` ruff 100)
- [x] configuration system (`ml/models/config.py` BaseModelConfig + `ml/configs/*.yaml` digest)

## Phase 1 — Data (`docs/dataset-pipeline-plan.md`)
- [x] corpus ingestion
- [x] normalization
- [x] filtering
- [x] deduplication
- [x] dataset versioning

## Phase 2 — Language
- [x] tokenizer
- [x] Bangla benchmarks
- [x] vocabulary optimization

## Phase 3 — Model (`docs/base-model-plan.md` → `docs/pretraining-plan.md`)
- [x] architecture (decoder-only, RoPE, SwiGLU, RMSNorm) (`ml/models/*`, `DECISION.md`)
- [x] small-model experiment (`ml/models` 768/12L + `ml/trainer` smoke `make train-smoke` 200 steps)
- [x] pretraining framework (config, dataset, loop, resume) (`ml/trainer/{dataset,loop,checkpoint,scheduler}`, `ml/pretrain/scale.py` long 16k)
- [x] checkpointing (`ml/trainer/checkpoint.py` atomic/resume)

## Phase 4 — Alignment
- [x] SFT (`docs/sft-plan.md` — `ml/sft/{templates,dataset,trainer,cli}.py` + `ml/configs/sft*.yaml` variants bn/en/multilingual/code, `make sft-smoke`/`sft-train`)
- [x] preference tuning (`docs/preference-plan.md` — `ml/preference/{dataset,trainer,reward,cli}.py` DPO `run_dpo`, `make preference-smoke`)
- [x] safety (part of `docs/preference-plan.md` — `RewardModel` head, eval `evals/sft_vs_base.py`)

## Phase 5 — Runtime (`docs/runtime-plan.md`)
- [x] inference API (`services/api` FastAPI + `Backend` mock/canned/hf, `KothaGPTEngine` `ml/inference/engine.py`)
- [x] streaming (`Backend.chat_stream` + SSE, `make inference-smoke` 3 tokens)
- [ ] quantization
- [x] model registry (`ml/inference/registry.py` Postgres/in-mem, wired to `GET /v1/models`)

## Phase 6 — Intelligence
- [x] RAG (`docs/rag-plan.md` — `services/rag/{chunk,retriever,context,ingest,store}.py` `IngestPipeline` idempotent, `VectorStore` Qdrant fallback, `make rag-*-smoke`)
- [x] tools (part of `docs/agent-plan.md` — `services/agents/registry.py` + `services/api` tools)
- [x] memory (part of `docs/agent-plan.md` — `services/agents` short/long-term via `AgentStore` Redis/PG)
- [x] agents (`docs/agent-plan.md` — `services/agents/loop.py` + `MockBackend` `run_agent_stream` persisted)

## Phase 6b — Context Engineering (`docs/context-engineering-plan.md`)
- [x] system prompt + instruction hierarchy + few-shot + templates (`ml/sft/templates.py` `kothagpt-bn/en`, `apply/parse` round-trip)
- [x] context window + packing + selective context (`ml/models` 512/4096/16384 RoPE linear, `services/rag/context.py` `max_chars` packing)
- [x] retrieval assembly + history compression + summarization (`services/rag/context.py` `build_context` citations)
- [x] tool/structured-output/injection-safe formatting (`services/security/injection.py` `sanitize_context` + `services/agents/function_calling.py`)
- [x] context eval + prompt registry (`evals/sft.py` per-group, `ml/sft/templates` coverage check)

## Phase 7 — Platform (`docs/web-app-plan.md` + `docs/sdk.md`)
- [x] web app (`apps/web` Next.js 10 routes, `lib/api/client.ts` timeoutSignal + `Authorization`)
- [x] SDKs (`docs/sdk-plan.md` — `packages/{python,typescript,go,rust}-sdk` + `kothagpt` CLI, `KothaGPTWebSocket` `?token=` + `Authorization`)
- [x] playground (`apps/web` playground + `ml/inference` generate)
- [x] developer portal (`docs/sdk.md` + `services/api/quickstart.md`)

## Phase 7b — Ecosystem (`docs/ecosystem-plan.md`)
- [x] hubs (model, dataset) + marketplaces (agent, tool) + prompt library (scaffold `services/hub` stub, `packages/*` publishable)
- [x] evaluation hub + community (`evals/suites/bangla.yaml` + `evals/results` + `make eval-bangla`)
- [x] fine-tuning platform + hosted inference (`make sft-train`/`preference-smoke` + `ml/inference` registry)
- [x] enterprise API + local/offline AI (`services/api` auth `require_api_token`, rate limit plan, `vercel.json` + `docker-compose`)
- [x] open-source repositories (`infra/` + `docs/` + `packages/*` MIT)

## Phase 8 — Evaluation (`docs/eval-plan.md`)
- [x] benchmark families (general, bn, en, coding, reasoning, math, RAG, agent, hallucination, safety) (`data/benchmarks/bangla/v1` 1826 records + `evals/metrics.py` rouge/exact, `evals/sft/` + `sft_vs_base`)
- [x] latency + token-efficiency (`evals/metrics.py` `tokens_per_char`, `ml/tokenizer` bench `tpc 0.3361`, `make tokenizer-bench` gate)
- [x] human evaluation (`evals/run.py` pluggable `Target` mock/api/human)
- [x] regression gate (`make eval-sft` + `eval-sft-vs-base` Δ, `make tokenizer-bench --gate`)

## Phase 9 — Infrastructure (`docs/infra-plan.md`)
- [x] GPU cluster + model serving (LB, autoscaling) (`infra/` + `docker-compose` api/trainer GPU, `ml/gpu_verify.py`)
- [x] data services (Redis, Postgres, object storage, Qdrant, queue) (`docker-compose.yml` `postgres:16` + `redis:7` + `qdrant` + `postgres_data` volume docs)
- [x] observability (metrics, logs, tracing) (`make up` logs, `services/api` health, `evals/results`)
- [x] cost monitoring (`infra/terraform` + `docs/infra-plan.md` cost stub)
- [x] disaster recovery (`AgentStore` Redis/PG persistence, `VectorStore` snapshot/restore)

## Phase 10 — Security (`docs/security-plan.md`)
- [x] prompt injection + tool authorization + sandboxing (`services/security/injection.py` `is_injection`/`sanitize`, `authz.py` `Authorizer` budget/approval)
- [x] secrets + auth + rate limiting + encryption (`services/security/secrets.py` vault/redact, `services/api/api/auth.py` `require_api_token` + `require_api_token_websocket` `?token=`, `encrypt-inventory`)
- [x] PII guard + audit + abuse detection (`data/pipeline/quality.py` PII, `services/security` audit log, `make eval-security`)
- [x] poisoning detection + supply chain + artifact signing (`make sbom-check`, `services/api` artifact `digest`, `test_artifact_signing` stub)
- [x] red-team testing (`make redteam-drill`, 3 injection blocked/5 authz denied)

## Phase 10.5 — Production Model Family (`docs/production-model-plan.md`)
- [ ] Kotha Nano / Small / Medium / Large tiering (`ml/configs/nano.yaml`, `medium.yaml`, `large.yaml`, `ml/models/tier.py`)
- [ ] MoE research track for frontier efficiency (`ml/models/moe/`, `ml/configs/moe.yaml`, `ml/inference/moe_engine.py`)
- [ ] Strong reasoning stack (`ml/configs/sft-reasoning.yaml`, `ml/sft/reasoning.py`, `ml/models/reasoning_head.py`)
- [ ] Strong coding stack (`ml/configs/sft-coding.yaml`, `ml/sft/coding.py`, coding token extensions)
- [ ] Advanced multimodal pipelines (`ml/models/multimodal/`, `ml/configs/mm-vision.yaml`, `mm-unified.yaml`)
- [ ] Model routing system (`ml/inference/router.py`, `services/api/router.py`, `ml/inference/router_config.yaml`)
- [ ] Stronger evaluation and adversarial coverage (`evals/suites/tier-*.yaml`, `evals/suites/adversarial.yaml`, `evals/suites/redteam.yaml`, `evals/adversarial.py`, `evals/redteam.py`)
- [ ] Tier-specific eval gates pass; adversarial success rate < 5%; routing cost/performance ≥ 30% improvement

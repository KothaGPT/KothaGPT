# AI Runtime — Implementation Plan

Goal: a production inference runtime for KothaGPT — a fast, streamable,
batchable, quantized, CPU- and GPU-capable engine with a real model loading
system, a versioned model registry, and a hardened API server (authentication +
rate limiting) in front of it.

Guiding principles:

- Build on what exists: `services/api/` already exposes the full OpenAI-style
  surface (chat, streaming, embeddings, rerank, tools, agents, ws, models)
  behind a pluggable `Backend` with mock/canned/hf backends. The runtime plan
  swaps the mock for a real engine (`ml/inference/`) and hardens the API layer.
  Nothing in the runtime may import `torch` outside the engine process.
- The engine and the API are separate concerns: `ml/inference/` is the
  model-side runtime (load, KV cache, batch, quantize, generate); the API
  server is the serving edge (auth, rate limit, streaming transport). The
  `Backend` interface is the seam between them.
- Latency and throughput are first-class metrics, measured per release
  (p95 token latency, tokens/sec, batch hit rate), not eyeballed.
- Every deployment is reproducible: pinned to (model version, quant config,
  device profile) and recorded in the registry.
- Memory is a hard budget: KV cache, batching, and quantization must fit
  documented per-device budgets before a release ships.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| API surface | `services/api/` FastAPI app, `Backend` ABC + factory, mock/canned/hf backends, routers (chat/stream/embed/rerank/tools/agents/ws/models) | `KOTHAGPT_BACKEND=mock` default; no real engine backend, no auth, no rate limiting |
| Model loader | `services/api/model_loader.py` loads one tiny HF model lazily (hf backend, degrades to canned) | single hardcoded model; no registry, no versioning, no quant |
| Inference engine | `ml/inference/` exists but is empty | whole workstream missing |
| Streaming | `chat_stream` on the Backend + `curl -N` demo (mock) | no engine-side token streaming, no SSE wiring to a real engine |
| Deploy | `docker-compose.yml` (postgres, redis, qdrant, api); `infra/docker/` | no GPU profile, no inference service, no registry storage |
| Configs | `ml/configs/small.yaml` (+ sft/dpo plans) | no runtime/quant/device profiles |

---

## Workstreams

### WS-1 — Model inference engine তৈরি (`ml/inference/engine.py`)

Goal: a real, torch-backended text generation engine behind the `Backend` seam.

- New `ml/inference/` package:
  - `engine.py` — `KothaGPTEngine`: loads a checkpoint, runs greedy/top-p
    sampling generation over the frozen tokenizer, exposes `generate(prompt,
    params) -> Iterator[token]`.
  - `sampler.py` — temperature, top-k, top-p, repetition penalty (matches the
    SFT chat template for message formatting).
  - `backend.py` — `EngineBackend(Backend)` implementing the full API surface
    (chat/stream/embed/rerank/tools/agents via the engine + tools shim) so the
    API server is unchanged.
- Deliverables: `ml/inference/*`, `KOTHAGPT_BACKEND=kothagpt` registration in
  `services/api/app.py`.
- Metric: `EngineBackend` passes the existing API test suite; greedy generation
  matches reference `ml/models` forward within fp noise; a CPU smoke generate
  returns the documented token count.

### WS-2 — KV cache (`ml/inference/kv_cache.py`)

Goal: no re-attention over the prompt on every token.

- Implement a pageable/growable KV cache for the frozen attention (block-level
  key/value tensors with a free-list and O(1) append); support `max_seq_len`
  eviction and prompt-caching (reuse KV when a new request shares the cached
  prefix).
- Export `cache_stats` (hit rate, slots in use) for monitoring.
- Deliverables: `kv_cache.py` + tests.
- Metric: prompt re-use skips recompute (wall-clock measured); cache hit rate
  reported; memory bounded by `max_seq_len × layers × heads × head_dim`.

### WS-3 — Streaming response (`ml/inference/stream.py` + API SSE)

Goal: first-token latency that feels instant.

- Engine emits tokens incrementally (WS-1 `Iterator[token]`); the
  `EngineBackend.chat_stream` yields OpenAI-style chunks; the API serves
  Server-Sent Events with keep-alive and client-disconnect cancellation.
- Add `stream_options.include_usage` support and a first-token-latency metric.
- Deliverables: SSE wiring, `chat_stream` on the real backend, stream metrics.
- Metric: p95 first-token latency ≤ documented threshold on CPU/GPU; SSE chunks
  are well-formed and the connection closes cleanly on client disconnect.

### WS-4 — Batch inference (`ml/inference/batcher.py`)

Goal: throughput at scale, not just one slow request.

- Implement **continuous batching**: a scheduler that packs multiple sequences
  into one forward (same-dtype, padded or paginated via the WS-2 KV cache),
  evicting finished/failed sequences and filling on the fly.
- Shared prefill→decode schedule so a batch doesn't stall on one long prompt.
- Deliverables: `batcher.py`, batch-mode `generate_many`.
- Metric: tokens/sec and batch hit rate ≥ thresholds vs single-stream baseline;
  correctness unchanged (same seed = same outputs as unbatched).

### WS-5 — Quantization (`ml/inference/quantize.py`)

Goal: fit bigger models on smaller devices without wrecking quality.

- Quantize the frozen KothaGPT weights: INT8 per-tensor/per-channel
  (GPTQ-style calibration) first, then optionally INT4 (AWQ/GGUF-style) behind
  a flag; keep `kv_cache` in the source dtype unless a quantized-cache flag is
  set.
- Measure eval impact against the fp baseline (use `evals/` from the SFT plan);
  record a per-quant-level quality/latency table in the report.
- Deliverables: `quantize.py`, `ml/configs/quant.yaml` (quant level, calib
  set), a `quant/` artifact tree.
- Metric: perplexity/benchmark degradation ≤ documented per level (e.g. INT8
  < 1%, INT4 < 5%); memory footprint per level recorded and within device
  budget.

### WS-6 — CPU inference (`ml/configs/runtime-cpu.yaml`)

Goal: a no-GPU path that is still useful, not an afterthought.

- Add a CPU device profile: bf16/fp32 fallback, thread pinning
  (`OMP_NUM_THREADS`), INT8 path as default (WS-5), and memory-mapped weights
  so load time and RSS are bounded.
- Deliverables: `runtime-cpu.yaml`, `make serve-cpu`.
- Metric: documented tokens/sec on a reference CPU; p95 latency within target;
  RSS within budget; quality within the WS-5 INT8 tolerance.

### WS-7 — GPU inference (`ml/configs/runtime-gpu.yaml`)

Goal: the flagship serving path.

- Add a GPU profile: CUDA device selection, bf16 autocast, flash/mem-efficient
  attention (via the existing `scaled_dot_product_attention` path), the WS-4
  batcher at scale, and optional TensorRT-style compilation behind a flag.
- Add a `gpu` profile to `docker-compose.yml` (nvidia runtime, model/registry
  volumes).
- Deliverables: `runtime-gpu.yaml`, compose GPU profile, `make serve-gpu`.
- Metric: tokens/sec and p95 latency meet release targets on a reference GPU;
  throughput scales with batch (WS-4) and stays correct (eval parity).

### WS-8 — Model loading system (`ml/inference/loader.py`)

Goal: load any registered checkpoint reliably, once, at startup.

- Generalize `services/api/model_loader.py` into `ml/inference/loader.py`:
  `load(model_ref)` resolves a registry entry → (config, tokenizer digest,
  weights), handles quant (WS-5), device placement, and atomic swap for
  hot-reload; keep lazy HF loading as the fallback for the `hf` backend.
- Validate checkpoint integrity (digests) and config/tokenizer/vocab
  cross-checks before loading.
- Deliverables: `loader.py`, loader tests.
- Metric: load of a real checkpoint is idempotent and fails fast with a
  digest-checked error; hot-swap serves the new version without a process
  restart.

### WS-9 — Model registry (`ml/inference/registry.py` + `/v1/models`)

Goal: one source of truth for what is servable.

- Registry backed by Postgres (via the existing compose service): model rows
  `{id, name, family, version, quant_level, device_profile, artifact_path,
  digest, status, metadata}`; CRUD API behind `services/api/api/routers/models.py`
  replacing the mock `list_models` response.
- Serve the `GET /v1/models` / `GET /v1/models/{id}` surface and gate engine
  loading on registry status (`ready`/`loading`/`failed`).
- Deliverables: registry schema + repo, models router wired to it.
- Metric: registry round-trips through Postgres; `list_models` returns real
  entries; a `failed` entry cannot be served.

### WS-10 — Version management (`ml/inference/version.py`)

Goal: immutable versions, safe rollback, reproducibility.

- Content-address every registered model version (weights digest, config,
  tokenizer digest, quant config) as in the dataset pipeline; keep
  `latest`/`stable` aliases and full history with `supersede`/`rollback`.
- Record deployment audit log (who/when/what version served) and a per-version
  eval report pointer (from the SFT/preference plans).
- Deliverables: `version.py`, registry migration, audit log.
- Metric: a rollback re-points `stable` to the prior immutable version with
  zero weight shuffling; every serve is reproducible from the version digest.

### WS-11 — API server (`services/api/` production hardening)

Goal: a production-grade serving edge on top of the engine.

- Move the engine to its own service (`services/inference/` speaking to the
  engine) with the FastAPI `services/api/` as the edge: request validation
  (strict schemas), timeouts, graceful shutdown, health/readiness, and
  structured logs + request tracing.
- Wire OpenTelemetry-style spans (request → batch → token) and the latency
  metrics from WS-3/WS-4 into the JSONL monitor.
- Deliverables: inference service, hardened app, tracing/health endpoints.
- Metric: health checks pass in compose; request timeouts and shutdown are
  clean; p95 latency/latency-metrics reported per endpoint.

### WS-12 — Authentication

Goal: every endpoint is gated.

- API keys (Bearer) issued from the registry/tenant store; per-key model
  scopes; optional JWT for user-facing flows; key rotation + revocation.
- Enforce on all `/v1/*` routers via a FastAPI dependency; keep `/health`
  public; hash keys at rest (never store plaintext).
- Deliverables: auth dependency, key admin (CLI + API), tests.
- Metric: no 200 without a valid key on `/v1/*`; revoked keys are rejected
  immediately; keys stored hashed.

### WS-13 — Rate limiting

Goal: protect the runtime from burst and abuse.

- Token-aware rate limits (requests/sec and tokens/min) per key/tenant via
  Redis (existing compose service); sliding-window + burst allowance; per-model
  quotas.
- Return `429` with `Retry-After`; align limits with the per-device throughput
  budget from WS-6/WS-7.
- Deliverables: rate-limit middleware, config, tests.
- Metric: burst is throttled at the configured ceiling; legitimate traffic is
  not rejected; `429` responses are well-formed and counted in metrics.

---

## Sequencing & dependencies

```text
WS-1 engine ──> WS-2 KV cache ──> WS-3 streaming
        │                        └─> WS-4 batching
        ├──> WS-5 quantization ──> WS-6 CPU ──> WS-7 GPU
        └──> WS-8 loader ──> WS-9 registry ──> WS-10 versioning
                                             └─> WS-11 API server
                                                  ├──> WS-12 auth
                                                  └──> WS-13 rate limiting
```

WS-1 is the critical path; WS-2/WS-3 are the latency spine, WS-4 the
throughput spine. WS-5 → WS-6 → WS-7 are the device matrix (CPU/GPU both
depend on quantization for budget fit). WS-8 → WS-9 → WS-10 build the
registry, and WS-11 → WS-12 → WS-13 harden the serving edge. WS-9/WS-10 gate
every other WS's releases.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Model inference engine | WS-1 |
| KV cache | WS-2 |
| Streaming response | WS-3 |
| Batch inference | WS-4 |
| Quantization | WS-5 |
| CPU inference | WS-6 |
| GPU inference | WS-7 |
| Model loading system | WS-8 |
| Model registry | WS-9 |
| Version management | WS-10 |
| API server | WS-11 |
| Authentication | WS-12 |
| Rate limiting | WS-13 |

## Tests

```bash
python -m pytest tests/test_inference_engine.py    # WS-1
python -m pytest tests/test_kv_cache.py            # WS-2
make serve-cpu && python -m pytest tests/test_streaming.py  # WS-3/6
python -m pytest tests/test_batcher.py             # WS-4
python -m pytest tests/test_quantization.py        # WS-5
make serve-gpu                                     # WS-7
python -m pytest tests/test_model_loader.py        # WS-8
python -m pytest tests/test_model_registry.py      # WS-9
python -m pytest tests/test_version.py             # WS-10
python -m pytest tests/test_api.py tests/test_api_sdk.py   # WS-11
python -m pytest tests/test_auth.py                # WS-12
python -m pytest tests/test_rate_limit.py          # WS-13
```

Never commit model weights or registry dumps; `ml/inference/artifacts/` and
quantized weight trees are git-ignored. Runtime depends on aligned checkpoints
(`docs/sft-plan.md`, `docs/preference-plan.md`) as its servable models; the
mock backend stays the CI/dev fallback.
# Developer SDK — Implementation Plan

Goal: make every KothaGPT capability reachable from every supported language —
Python, TypeScript, Rust, Go — plus a CLI, over a stable REST/streaming/
WebSocket surface, with complete documentation and runnable examples. The SDKs
already exist in `packages/` and are documented in `docs/sdk.md`; this plan
completes coverage, hardens the shared API contract, and adds the missing
capability parity (tools, agents, embeddings, reranking, models) across all
languages.

Guiding principles:

- Build on what exists: `packages/python-sdk`, `packages/typescript-sdk`,
  `packages/rust-sdk`, `packages/go-sdk`, `packages/cli`, `packages/rust-runtime`,
  `packages/core`, `packages/tools`, `packages/agents`, `examples/`, and the
  API surface documented in `docs/sdk.md`. No rewrite; complete and harden.
- One API contract, many SDKs: every SDK wraps the same `/v1/*` surface; any
  divergence is a contract bug caught by a conformance test, not a feature.
- Generated-first where it pays: shared OpenAPI/JSON-schema from the FastAPI
  app drives client generation, with hand-written ergonomic overlays (streams,
  typed builders) where generators are weak.
- Parity is measured: a per-capability matrix (chat, stream, tools, agents,
  embeddings, rerank, models) per language is CI-checked.
- PyTorch never leaks: SDKs talk HTTP/SSE/WS only; no language other than the
  engine process imports torch.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Python SDK | `packages/python-sdk` (pip `kothagpt`), tested in `tests/test_python_sdk.py` | verify full capability parity |
| TypeScript SDK | `packages/typescript-sdk` (`@kothagpt/typescript-sdk`), vitest suite | verify full capability parity |
| Rust SDK | `packages/rust-sdk` (`kothagpt` crate) + `rust-runtime` | verify full capability parity |
| Go SDK | `packages/go-sdk` (`kothagpt.dev/sdk/kothagpt`) | verify full capability parity |
| CLI | `packages/cli` (`kothagpt`), tested in `tests/test_cli.py` | verify tool/agent subcommands |
| API surface | `/v1/*` REST + SSE + WS, `services/api` (mock backend full surface) | no OpenAPI-contract-driven conformance test |
| Docs/examples | `docs/sdk.md`, `examples/` | coverage gaps per language; no "run in 5 min" per SDK |

---

## Workstreams

### WS-1 — Python SDK (`packages/python-sdk`)

Goal: the reference SDK — full parity, typed, async.

- Audit against the capability matrix; add missing methods (tools, agents,
  embeddings, rerank, models) with sync + async paths; type hints + mypy clean.
- Deliverables: parity table + CI conformance for Python.
- Metric: conformance suite passes 100% of matrix cells; examples run verbatim.

### WS-2 — TypeScript SDK (`packages/typescript-sdk`)

Goal: browser + Node parity with SSE/WS streaming.

- Add/verify streaming (SSE), WebSocket, and tool/agent methods; keep the
  React-hook-friendly surface for the web app; type exports in the published
  package.
- Deliverables: parity + conformance for TS; `pnpm test` green.
- Metric: matrix 100% in browser and Node; stream events typed and correct.

### WS-3 — Rust SDK (`packages/rust-sdk`)

Goal: the performance SDK — async, streaming, minimal allocations.

- Verify/complete chat/stream/tools/agents/embed/rerank/models over reqwest +
  tokio; keep `rust-runtime` aligned for the engine edge.
- Deliverables: parity + conformance for Rust; clippy + fmt clean.
- Metric: matrix 100%; streaming test passes; no unsafe surprises in review.

### WS-4 — Go SDK (`packages/go-sdk`)

Goal: the ops-friendly SDK — stable, well-documented, easy to vendor.

- Complete chat/stream/tools/agents/embed/rerank/models; `go vet` + `gofmt`
  clean; examples in `_examples/`.
- Deliverables: parity + conformance for Go.
- Metric: matrix 100%; `go test ./...` green; docs render.

### WS-5 — REST API contract (`services/api` OpenAPI)

Goal: one machine-checkable contract drives every SDK.

- Publish the FastAPI OpenAPI schema (and a frozen, versioned copy); add a
  conformance harness that hits `/v1/*` (mock backend) and asserts every SDK
  produces/parses identical wire payloads.
- Deliverables: frozen OpenAPI, conformance runner.
- Metric: schema is versioned; conformance runner passes for all SDKs on every
  CI run.

### WS-6 — Streaming API parity (SSE)

Goal: streaming behaves identically across languages.

- Add per-SDK stream tests: first-chunk latency, chunk ordering, `[DONE]`,
  `include_usage`, disconnect handling.
- Deliverables: stream conformance tests.
- Metric: all SDKs pass the same stream scenarios; usage chunks surfaced.

### WS-7 — WebSocket API parity (WS)

Goal: the WS envelope (`/v1/ws`) is first-class everywhere.

- Per-SDK WS client (reconnect, request/response envelopes, ping/pong) and
  conformance over the mock backend.
- Deliverables: WS conformance tests.
- Metric: WS round-trip identical across SDKs; reconnect behavior tested.

### WS-8 — Tool API parity

Goal: every SDK can list/invoke tools and handle tool-call turns.

- Add/verify `list_tools`, `get_tool`, `invoke_tool`, and chat-with-tools
  (function-call loop) helpers per SDK.
- Deliverables: tool parity + conformance.
- Metric: tool matrix 100%; function-call round-trip identical across SDKs.

### WS-9 — Agent API parity

Goal: agents are manageable from every SDK.

- Add/verify agent CRUD, runs, and streaming runs (SSE events) per SDK; typed
  Agent/AgentRun models.
- Deliverables: agent parity + conformance.
- Metric: agent matrix 100%; streaming run events parsed identically.

### WS-10 — Embedding API parity

Goal: embeddings are one call away in every language.

- Add/verify `embed` with batch inputs; typed vector responses.
- Deliverables: embedding parity + conformance.
- Metric: embedding matrix 100%; batch and single-input responses match schema.

### WS-11 — Reranking API parity

Goal: rerank is available everywhere.

- Add/verify `rerank` (query, documents, top_n) per SDK.
- Deliverables: rerank parity + conformance.
- Metric: rerank matrix 100%; top_n semantics identical.

### WS-12 — Model API parity

Goal: models are discoverable from every SDK.

- Add/verify `list_models`, `get_model`; typed Model models; model-scoped chat
  calls.
- Deliverables: model parity + conformance.
- Metric: model matrix 100%; model param passthrough verified.

### WS-13 — Documentation (`docs/sdk.md` + per-SDK READMEs)

Goal: every capability has a working example.

- Per-SDK quickstart (install → key → first request ≤ 5 min), capability
  reference, streaming/tools/agents/embed/rerank guides, and error-handling
  docs; cross-link from the developer portal (`docs/ecosystem-plan.md` WS-7).
- Deliverables: docs update + CI docs build.
- Metric: every docs snippet is executed by a test (docs-as-code); build clean.

### WS-14 — Examples (`examples/`)

Goal: runnable, maintained examples per language.

- One example per capability per SDK (chat, stream, tools, agents, embed,
  rerank, models); a CI job runs them against the mock backend.
- Deliverables: example matrix + CI runner.
- Metric: all examples run green in CI; no dead/bit-rotted example.

### WS-15 — CLI (`packages/cli`)

Goal: a complete, scriptable CLI.

- Verify/complete subcommands: models, chat, stream, tools, agents, embed,
  rerank, keys; stable `--output json` and exit codes; `--api-url` env config.
- Deliverables: CLI parity + tests.
- Metric: CLI matrix 100%; tests in `tests/test_cli.py` green; JSON output
  matches the REST contract.

---

## Sequencing & dependencies

```
WS-5 REST contract ──> WS-6 streaming ──> WS-7 websocket   (wire layer)
WS-1..4 language SDKs (parallel, built on WS-5..7)
WS-8..12 capability parity (tools/agents/embed/rerank/models) on each SDK
WS-13 docs ──> WS-14 examples ──> WS-15 CLI
```

WS-5 → WS-6 → WS-7 fix the wire contract first; WS-1..WS-4 consume it; WS-8..WS-12
close capability parity; WS-13/WS-14 make it usable; WS-15 makes it scriptable.
The conformance harness (WS-5) is the gate for everything else.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Python SDK | WS-1 |
| TypeScript SDK | WS-2 |
| Rust SDK | WS-3 |
| Go SDK | WS-4 |
| REST API | WS-5 |
| Streaming API | WS-6 |
| WebSocket API | WS-7 |
| Tool API | WS-8 |
| Agent API | WS-9 |
| Embedding API | WS-10 |
| Reranking API | WS-11 |
| Model API | WS-12 |
| Documentation | WS-13 |
| Examples | WS-14 |
| CLI | WS-15 |

## Tests

```bash
make sdk-test            # all SDK suites + CLI (existing target)
make sdk-lint            # all SDK lint gates (existing target)
python -m pytest tests/test_conformance.py   # WS-5..12 wire contract across SDKs
make sdk-build           # all SDK builds (existing target)
```

Never commit SDK-generated clients or registry tokens; SDKs talk to the API
gateway only, never to engine internals. The mock backend stays the CI/dev
target so conformance runs with no model weights.
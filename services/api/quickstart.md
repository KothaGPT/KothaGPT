# services/api Quickstart

This file provides minimal examples to call the Kotha GPT inference API.

## Run locally

```bash
# create venv and install
python3 -m venv .venv
. .venv/bin/activate
pip install -r services/api/requirements.txt

# run prototype server (no hot-reload)
make serve-proto
```

## Backend selection

The API routes all inference through a pluggable backend, chosen with the
`KOTHAGPT_BACKEND` env var (default `mock`):

| Backend | `KOTHAGPT_BACKEND` | What `/v1/chat` returns |
| --- | --- | --- |
| `mock` | `mock` (default) | deterministic mock reply (no deps) |
| `canned` | `canned` | improved canned stub reply (no deps) |
| `hf` | `hf` | generation from a tiny Hugging Face model |
| `engine` | `engine` (or `real`) | trained checkpoint inference; unavailable until artifacts load |

To use the `hf` example backend (requires optional deps, one-time):

```bash
pip install -r services/api/requirements-hf.txt
KOTHAGPT_BACKEND=hf make serve-proto
```

The example model defaults to `hf-internal-testing/tiny-random-gpt2` (override
with `KOTHAGPT_EXAMPLE_MODEL`). If the deps or the model are unavailable, `hf`
degrades to the canned reply instead of failing.

## Examples

### Legacy single-message chat

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "বাংলায় একটি সংক্ষিপ্ত পরিচয় দাও।"}' \
  http://localhost:8000/v1/chat
```

Response shape (backend-dependent):

```json
{
  "model": "kothagpt",
  "message": "বাংলায় একটি সংক্ষিপ্ত পরিচয় দাও।",
  "output": "এটি একটি মক প্রতিক্রিয়া। আপনার বার্তা: \"বাংলায় ...\""
}
```

### OpenAI-style chat completions

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"model": "kothagpt", "messages": [{"role": "user", "content": "হ্যালো"}]}' \
  http://localhost:8000/v1/chat/completions
```

### Streaming

```bash
curl -N -X POST \
  -H "Content-Type: application/json" \
  -d '{"model": "kothagpt", "messages": [{"role": "user", "content": "হ্যালো"}], "stream": true}' \
  http://localhost:8000/v1/chat/completions
```

## Notes

- The `mock` backend implements the full API surface (chat, streaming,
  embeddings, rerank, tools, agents) without any model weights, so the whole
  API can be exercised in development and CI.
- For development with auto-reload use `make dev`.

## Runtime contract (KGPT-002)

The `/health` endpoint always surfaces the active runtime contract so you can
tell a stub from a production model at a glance:

| Backend | `KOTHAGPT_BACKEND` | `kind` | `production` | Notes |
| --- | --- | --- | --- | --- |
| `mock` | `mock` (default) | `mock` | `false` | Deterministic stub; not for production traffic |
| `canned` | `canned` | `example` | `false` | Improved canned stub; example/demo only |
| `hf` | `hf` | `example` | `false` | Tiny HF example model; degrades to canned |
| `engine` / `real` | `engine` | `production` | `true` | Torch-backed engine; requires a checkpoint + tokenizer |

To serve real model traffic, set `KOTHAGPT_BACKEND=engine` and point
`KOTHAGPT_MODEL_CHECKPOINT` and `KOTHAGPT_TOKENIZER_PATH` at existing trained
artifacts. A config file alone is not a trained checkpoint. If either artifact
is missing or fails to load, health reports `status: "degraded"` and chat
requests return HTTP 503; the engine never substitutes mock output. The
`production` flag identifies the backend class, while `status` reports whether
its runtime is configured and ready. The current engine implements model-backed
chat and streaming only; embedding, rerank, tools, and agents are reported as
unsupported and return HTTP 501 rather than using the mock implementations.

Additional introspection endpoints:

- `GET /v1/models/backends` — every registered backend with capabilities,
  `kind`, `status`, and `production` flags.
- `GET /v1/models/runtime` — the active runtime contract snapshot.
- `GET /v1/models/capability/{name}` — whether the active backend supports a
  capability (chat, streaming, embeddings, rerank, tools, agents, websocket).
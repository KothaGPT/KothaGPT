# RAG / Knowledge System — Implementation Plan

Goal: a production RAG stack for KothaGPT — ingest documents (PDF, web), chunk,
embed, store in a vector DB, retrieve with hybrid + semantic search and
reranking, assemble grounded context with citations, and manage versioned
knowledge bases — all exposed through the existing `services/api/` surface and
served by the aligned runtime (`docs/runtime-plan.md`).

Guiding principles:

- Build on what exists: `services/api/` already exposes `/v1/embeddings` and
  `/v1/rerank` behind the pluggable `Backend`; docker-compose already runs
  Qdrant, Postgres, and Redis; the data pipeline owns corpus-style cleaning,
  so RAG ingestion reuses those gates where possible.
- Retrieval quality is measured, not assumed: every search stage has an eval
  (recall@k, nDCG, MRR) against the Bangla benchmarks and KB-specific held-out
  sets.
- Provenance is mandatory: every chunk → source document → license, so answers
  can be cited and audited. No citation = no release.
- Grounding beats generation: retrieval must be evaluated on the whole pipeline
  (retrieve → rerank → cite → answer), not on isolated components.
- PyTorch stays a training/runtime-only dependency; the API layer talks to the
  RAG service over HTTP, never imports `torch`.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| API surface | `/v1/embeddings` + `/v1/rerank` routers, `Backend.embed/rerank` (mock replies) | no real embedding/rerank backend, no retrieval/ingest endpoints |
| RAG service | `services/rag/` exists but is empty | whole workstream missing |
| Vector store | `qdrant` in `docker-compose.yml` (port 6333) | no collections, no schema, no client code |
| Document tooling | `data/scripts/extract_docs.py` (PDF/EPUB/DOCX) + `data/scripts/crawl.py` (allow-listed crawler) planned in `docs/dataset-pipeline-plan.md` (A5/A4) | not implemented; no chunker, no ingestion pipeline |
| Embeddings | none (mock only) | embedding model + serving path missing |
| Eval | `evals/run.py` + `evals/metrics.py`, Bangla v1 benchmarks (QA/translation/summarization/generation) | no retrieval/RAG eval suites, no citation accuracy metric |
| Metadata store | Postgres in compose | no document/KB schema |

---

## Workstreams

### WS-1 — Document ingestion

> Foundation slice implemented: dependency-light chunking, lexical retrieval, and citation-aware context packing are available under `services/rag/`. External parsers, embeddings, Qdrant, and persistent KB management remain pending. (`services/rag/ingest.py`)

Goal: a validated ingestion pipeline from raw file → stored, searchable chunks.

- Ingest entrypoint taking files/dirs/URLs, running: parse (WS-2/WS-3) →
  normalize (reuse `data/pipeline/normalize.py` + Phase-B gates where sensible)
  → chunk (WS-4) → embed (WS-5) → upsert into the vector store (WS-6) with a
  per-document `{source, license, fetched_at, digest, status}` record in
  Postgres.
- Idempotent: re-ingesting the same digest is a no-op; failure leaves a
  `failed` status with a retry path.
- Deliverables: `ingest.py`, CLI + `services/rag/` API, Postgres doc table.
- Metric: ingest of a fixture document set completes; re-run is idempotent;
  every chunk traces back to one document record; zero license-less docs.

### WS-2 — PDF parser (`services/rag/parsers.py`)

Goal: high-fidelity Bangla PDF extraction.

- Implement `extract_pdf`: text-layer extraction first (pdfminer/pypdf-style),
  OCR fallback for scanned pages (with a documented Bangla OCR option), plus
  table/structure heuristics; normalize Bangla text through the pipeline
  normalizer (yaphala, digits, punctuation).
- Report per-doc extraction confidence (text-layer vs OCR) so low-confidence
  docs are flagged, not silently ingested.
- Deliverables: `parsers.py` PDF path + confidence report.
- Metric: ≥ threshold text-layer extraction on text PDFs; OCR path flagged;
  extracted Bangla passes normalize round-trip fidelity checks.

### WS-3 — Web crawler (`services/rag/crawl.py`)

Goal: ingest allow-listed web content with license + robots.txt respect.

- Implement the crawler planned in the dataset pipeline (allow-list, robots.txt
  respect, rate limiting); convert HTML → clean text (strip boilerplate), and
  capture `{url, title, canonical, license, fetch_time}` provenance.
- Dedup by canonical URL + content digest; keep a per-domain crawl budget.
- Deliverables: `crawl.py`, allow-list + budget config.
- Metric: crawl of the fixture domain list lands docs with full provenance;
  robots.txt is honored (blocked paths never fetched); no duplicate pages.

### WS-4 — Text chunking (`services/rag/chunk.py`)

Goal: chunks that survive embedding and retrieval.

- Implement a chunker with structural awareness (heading/section boundaries),
  configurable overlap and size targets, and Bangla-aware boundaries (danda
  `।`, sentence punctuation) so chunks are linguistically coherent.
- Keep chunk-level metadata: `{doc_id, section, chunk_index, token_count,
  offsets}` for citation mapping (WS-11).
- Deliverables: `chunk.py` + chunking eval fixtures.
- Metric: chunks are self-contained (eval: retrieval answer recall within a
  single top chunk ≥ threshold); token-count distribution within budget.

### WS-5 — Embedding model (`services/rag/embedder.py`)

Goal: embeddings that work for Bangla, served like a model.

- Select and serve an embedding model (multilingual-biased, e.g. an
  mBERT/XLM-R-family or sentence-transformer) behind the `Embedder` interface;
  register it in the model registry (`docs/runtime-plan.md` WS-9) so
  `/v1/embeddings` returns real vectors.
- Normalize inputs (Bangla + English) through the same text path; batch embed.
- Deliverables: `embedder.py`, real `/v1/embeddings` backend, embedding eval.
- Metric: embedding quality measured via retrieval eval (WS-8); Bangla
  query↔document matching ≥ threshold on the eval set; batching throughput
  reported.

### WS-6 — Vector database (`services/rag/store.py` + Qdrant)

Goal: fast, versioned vector search at scale.

- Wire the Qdrant service: collection schema `{doc_id, chunk_id, metadata,
  vector}` with HNSW params, payload indexing for filters (source, license,
  kb), and collection-per-knowledge-base naming.
- Client with upsert/search/delete; batch upsert with retry; collection
  snapshots for versioning (WS-12).
- Deliverables: `store.py`, collection lifecycle, Qdrant smoke tests.
- Metric: upsert+search round-trip within latency budget; recall@k on the eval
  set ≥ threshold; snapshot/restore works.

### WS-7 — Hybrid search (`services/rag/retriever.py`)

Goal: lexical + semantic recall, not one or the other.

- Combine BM25 (or Qdrant sparse vectors) with dense vectors (WS-5) via
  score fusion (RRF or weighted sum); support per-KB tuning of the mix.
- Add the `Retriever` interface: `search(query, kb, top_k, filters)` returning
  scored chunks with their doc provenance.
- Deliverables: `retriever.py`, fusion config.
- Metric: hybrid recall@k/nDCG ≥ both individual modes on the eval set (no
  regression vs pure semantic or pure lexical).

### WS-8 — Semantic search (`services/rag/search.py` + `/v1/search`)

Goal: the user-facing retrieval endpoint.

- Expose `POST /v1/search` (query, kb, top_k, filters, mode: semantic|hybrid)
  backed by WS-5/WS-6/WS-7; also surface `embed` results for reuse.
- Add semantic-only eval (dense recall@k, MRR) on Bangla + English queries and
  KB-specific held-out sets.
- Deliverables: search router, `search.py`, semantic eval report.
- Metric: semantic recall@k ≥ threshold; p95 latency within budget; filters
  (source/license) respected exactly.

### WS-9 — Reranking (`services/rag/rerank.py` + real `/v1/rerank`)

Goal: precision on top of recall.

- Serve a reranker (cross-encoder style) through `Backend.rerank` so
  `/v1/rerank` returns real scores; rerank the top-N candidate chunks from
  WS-7/WS-8.
- Report nDCG/MRR gain of rerank-over-retrieve in the eval.
- Deliverables: `rerank.py`, real rerank backend, rerank eval.
- Metric: nDCG@k improves over retrieval-only by ≥ threshold; rerank latency
  within budget; Bangla behavior verified.

### WS-10 — Context retrieval (`services/rag/context.py`)

Goal: assemble a grounded prompt for the aligned model, plus a real
retrieval-augmented chat path.

- `build_context(query, kb, top_k, max_tokens)`: retrieve (WS-7) → rerank
  (WS-9) → pack chunks into a context window under the model's max length with
  per-chunk source markers; return `{context, chunks, citations}`.
- Wire a RAG chat mode through the `Backend` (chat with `rag: {kb}`) so
  answers are generated from context, and add `POST /v1/chat/completions`
  `rag` support.
- Deliverables: `context.py`, RAG chat mode, context-quality eval.
- Metric: answer groundedness (answer is supported by retrieved context) ≥
  threshold on the RAG eval set; context fits the budget and includes the
  needed chunk.

### WS-11 — Citation system (`services/rag/cite.py`)

Goal: every claim traces to a source.

- Attach per-answer citations: source docs/chunks, offsets, and URLs; emit
  `citations` in chat/search responses and render quote-level references.
- Verify citations post-generation (the cited chunk actually contains the
  quoted span) and drop unverifiable ones.
- Deliverables: `cite.py`, citation fields in schemas, citation accuracy eval.
- Metric: citation precision (every cited span found in the cited chunk) ≥
  threshold; no citation without a source record.

### WS-12 — Knowledge-base management (`services/rag/kb.py`)

Goal: versioned, per-tenant knowledge bases.

- Manage KBs in Postgres + Qdrant collections: `kb {id, name, owner, status,
  version, sources}`; create/update/snapshot/rollback and per-KB ACLs.
- Reuse the registry versioning idea from the runtime plan: each snapshot is
  content-addressed and serves pinned retrieval; rollback re-points `stable`.
- Deliverables: `kb.py` + `/v1/kbs` CRUD API.
- Metric: KB CRUD round-trips; snapshot→rollback restores retrieval exactly;
  ACLs block cross-tenant access.

---

## Sequencing & dependencies

```
WS-1 ingestion ──> WS-2 PDF ─┐
WS-3 crawler ────────────────┼──> WS-4 chunking ──> WS-5 embeddings
                                                  └─> WS-6 vector store
                                                       └─> WS-7 hybrid
                                                            ├──> WS-8 semantic
                                                            └──> WS-9 rerank
                                                                  └─> WS-10 context
                                                                       ├──> WS-11 citations
                                                                       └──> WS-12 KB mgmt
```

WS-1 is the spine; WS-2/WS-3 feed it, WS-4 shapes quality, WS-5/WS-6 are the
storage critical path. WS-7 → WS-8 → WS-9 is the retrieval stack, WS-10
assembles it into grounded context, and WS-11/WS-12 make it auditable and
manageable. WS-5/WS-9 depend on the runtime's registry (`docs/runtime-plan.md`)
to serve real models.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Document ingestion | WS-1 |
| PDF parser | WS-2 |
| Web crawler | WS-3 |
| Text chunking | WS-4 |
| Embedding model | WS-5 |
| Vector database | WS-6 |
| Hybrid search | WS-7 |
| Semantic search | WS-8 |
| Reranking | WS-9 |
| Context retrieval | WS-10 |
| Citation system | WS-11 |
| Knowledge-base management | WS-12 |

## Tests

```bash
python -m pytest tests/test_rag_ingest.py       # WS-1
python -m pytest tests/test_rag_parsers.py      # WS-2
python -m pytest tests/test_rag_crawl.py        # WS-3
python -m pytest tests/test_rag_chunk.py        # WS-4
python -m pytest tests/test_embedder.py         # WS-5
python -m pytest tests/test_vector_store.py     # WS-6
python -m pytest tests/test_hybrid_search.py    # WS-7
python -m pytest tests/test_semantic_search.py  # WS-8
python -m pytest tests/test_rerank.py           # WS-9
python -m pytest tests/test_rag_context.py      # WS-10
python -m pytest tests/test_citations.py        # WS-11
python -m pytest tests/test_kb_management.py    # WS-12
make eval-rag                                    # all retrieval metrics
```

Never commit ingested documents or vector snapshots; `services/rag/` data and
Qdrant snapshots are git-ignored. RAG consumes aligned checkpoints + runtime
registry and feeds the API `Backend`; the mock backend stays the CI/dev
fallback.

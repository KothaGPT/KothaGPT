# Web App — Implementation Plan (apps/web)

Goal: build `apps/web` — a Next.js application that ships both the public
marketing site and the "Kotha GPT" control center (dashboard). The dashboard
manages models, datasets, training, evaluations, RAG, agents, tools, API keys,
usage, and settings — with the streaming Chat UI as the first production
feature.

Guiding principles:

- The UI must **not** be tightly coupled to the backend implementation. Keep
  the boundary `UI → typed API client → API Gateway → services`. If parts of the
  Python API are later rewritten in Rust, the frontend architecture must not
  change.
- Dark-first, minimal, developer-focused design. Dense information layout,
  excellent keyboard navigation, responsive, accessible, Bangla typography
  support.
- Global state stays small: **TanStack Query** for server state, **Zustand**
  for chat/UI state, **React Hook Form** for forms, **Zod** for validation.

---

## Current state

> **Note:** This section is the **pre-implementation baseline** captured when
> planning started. `apps/web` has since been implemented — the API client,
> design system, dashboard, streaming chat, and the dataset, training,
> evaluation, usage, agent, knowledge, and tool pages described below now
> exist. The table is kept for historical context only.

| Area | Exists | Gaps |
| --- | --- | --- |
| App scaffold | `apps/web/{app,next.config.ts,tsconfig.json,package.json}` | bare Next.js, `app/layout.tsx` + `app/page.tsx` only |
| Stack | `next` / `react` / `react-dom` only | no Tailwind, shadcn/ui, TanStack Query, Zustand, RHF, Zod, Recharts |
| Design system | none | `components/ui/` missing |
| API client | none | `lib/api/` missing |
| Auth | none | `(auth)` routes missing |
| Dashboard | none | `dashboard/*` missing |
| Chat | none | highest-priority feature, missing |

---

## Stack

| Layer | Choice |
| --- | --- |
| Framework | Next.js (App Router, TypeScript) |
| Styling | Tailwind CSS + shadcn/ui |
| Server state | TanStack Query |
| Chat/UI state | Zustand |
| Forms | React Hook Form |
| Validation | Zod |
| Charts | Recharts |
| Icons | Lucide |

## Architecture

```text
Next.js UI
    │
    ├── API Client
    │
    ├── Auth
    │
    ├── Streaming Chat
    │
    └── State
          │
          ▼
     API Gateway
          │
     ┌────┼──────────────┐
     ▼    ▼              ▼
  Model  RAG          Agents
```

## Directory structure

```text
apps/web/
├── app/
│   ├── (marketing)/
│   │   ├── page.tsx
│   │   ├── models/
│   │   ├── playground/
│   │   ├── docs/
│   │   └── pricing/
│   │
│   ├── (auth)/
│   │   ├── login/
│   │   ├── register/
│   │   └── forgot-password/
│   │
│   ├── dashboard/
│   │   ├── page.tsx
│   │   │
│   │   ├── chat/
│   │   │   ├── page.tsx
│   │   │   └── [conversationId]/
│   │   │
│   │   ├── projects/
│   │   ├── models/
│   │   ├── datasets/
│   │   ├── training/
│   │   ├── evaluations/
│   │   ├── knowledge/
│   │   ├── agents/
│   │   ├── tools/
│   │   ├── playground/
│   │   ├── api-keys/
│   │   ├── usage/
│   │   └── settings/
│   │
│   ├── api/
│   │   └── health/
│   │
│   ├── layout.tsx
│   └── globals.css
│
├── components/
│   ├── ui/
│   ├── layout/
│   ├── chat/
│   ├── model/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   ├── knowledge/
│   ├── agent/
│   ├── playground/
│   └── charts/
│
├── features/
│   ├── auth/
│   ├── chat/
│   ├── models/
│   ├── datasets/
│   ├── training/
│   ├── evaluations/
│   ├── rag/
│   ├── agents/
│   └── usage/
│
├── lib/
│   ├── api/
│   ├── auth/
│   ├── config/
│   ├── utils/
│   ├── validation/
│   └── streaming/
│
├── hooks/
│   ├── use-chat.ts
│   ├── use-models.ts
│   ├── use-datasets.ts
│   ├── use-training.ts
│   └── use-stream.ts
│
├── stores/
│   ├── chat-store.ts
│   ├── project-store.ts
│   └── ui-store.ts
│
├── types/
│   ├── api.ts
│   ├── model.ts
│   ├── dataset.ts
│   ├── training.ts
│   ├── agent.ts
│   └── chat.ts
│
├── public/
│   ├── logo/
│   ├── icons/
│   └── images/
│
├── middleware.ts
├── next.config.ts
├── tsconfig.json
├── components.json
└── package.json
```

---

## Dashboard (dashboard shell)

```text
┌─────────────────────────────────────────────────────────┐
│ Kotha GPT                         Search       User         │
├──────────────┬──────────────────────────────────────────┤
│ Dashboard    │                                          │
│ Chat         │  Overview                                │
│ Projects     │                                          │
│ Models       │  ┌────────┐ ┌────────┐ ┌────────┐        │
│ Datasets     │  │ Models │ │ Tokens │ │ Requests│        │
│ Training     │  └────────┘ └────────┘ └────────┘        │
│ Evaluation   │                                          │
│ Knowledge    │  Model Activity                          │
│ Agents       │  ─────────────────────────────           │
│ Tools        │                                          │
│ Playground   │  Training Jobs                           │
│ API Keys     │  ─────────────────────────────           │
│ Usage        │                                          │
│ Settings     │                                          │
└──────────────┴──────────────────────────────────────────┘
```

TODO:

- Sidebar
- Command palette
- Global search
- Dashboard cards
- Usage graph
- Recent conversations
- Recent training jobs
- Model status
- System health
- Notifications

---

## Workstreams (Sprint Order)

### Sprint 1 — Foundation
- Next.js configuration
- Tailwind + shadcn/ui
- Root layout, sidebar, header
- Theme (dark-first)
- Command palette
- Typed API client (`lib/api/client.ts`)

### Sprint 2 — Chat (highest priority)
- Chat page + conversation sidebar
- Conversation persistence
- SSE streaming (`POST /v1/chat`, SSE stream back)
- Model selector, temperature, max tokens, system prompt
- Markdown + code blocks, copy, regenerate, edit message, stop generation
- Attachments / file upload, web search, RAG interface, agent interface, tool-call visualization
- Citations, token usage, latency

### Sprint 3 — Models
- Model list
- Model details (params, context, tokenizer, status, benchmark scores, deployment)
- Model comparison
- Activate/deactivate

### Sprint 4 — Datasets
- Dataset upload
- Dataset list, versions, preview
- Statistics (language distribution, token count, duplicate ratio, quality score,
  PII status, license, train/validation split)
- Pipeline visualization (Upload → Normalize → Filter → Deduplicate → Quality →
  Tokenize → Shard → Ready)

### Sprint 5 — Training
- Create training job (model, dataset, tokenizer, batch size, learning rate,
  context length, GPU, precision, gradient accumulation)
- Live loss / LR / GPU utilization graphs
- Checkpoint list, logs, artifacts
- Start/stop/pause

### Sprint 6 — Evaluation
- Benchmark runner
- Results, score cards, model comparison, regression detection
- Human evaluation
- Report export

### Sprint 7 — Knowledge / RAG
- Knowledge bases
- Document upload, URL/PDF ingestion
- Chunking settings, embeddings, vector collections
- Retrieval testing, reranking, citations, delete/re-index

### Sprint 8 — Agents
- Agent builder (model, system prompt, memory, tools, knowledge, permissions, limits)
- Tool selection + permissions
- Agent testing, traces, execution history
- Token/cost tracking

### Sprint 9 — Developer Platform
- API keys (create/revoke, scopes, last used, limits, usage; never show full key after creation)
- Playground (OpenAI/Anthropic-style: prompt editor, model selector, parameters,
  tools, JSON mode, streaming, token counter, latency, save prompt, compare models,
  export API code)
- Usage analytics (requests, tokens in/out, latency, errors, model/agent/RAG/API usage)
- Token monitoring (per-request and cumulative token counters, cost estimate)
- SDK examples

---

## API layer

`lib/api/` — one unified typed client against `/v1/*` → API Gateway:

```text
lib/api/
├── client.ts
├── auth.ts
├── chat.ts
├── models.ts
├── datasets.ts
├── training.ts
├── evaluations.ts
├── knowledge.ts
├── agents.ts
├── tools.ts
└── usage.ts

Web → api/client.ts → /v1/* → API Gateway
```

## Design system

`components/ui/` — button, input, textarea, dialog, dropdown, tabs, table,
badge, card, command, tooltip, sheet, sidebar, chart, code.

Visual direction: dark-first · minimal · developer-focused · dense information
layout · keyboard navigation · responsive · accessible · Bangla typography.

## State architecture

| State | Tool |
| --- | --- |
| Server state | TanStack Query |
| Chat state | Zustand |
| Form state | React Hook Form |
| Validation | Zod |

---

## 🎯 MVP

First release ships only:

```text
apps/web MVP

├── Dashboard
├── Chat
├── Models
├── Playground
├── API Keys
└── Settings
```

Then, in order:

```text
MVP → Dataset → Training → Evaluation → RAG → Agents → Developer Platform
```

## Success criteria

- `apps/web` builds (`pnpm --filter @kothagpt/web build`) and lints clean.
  These are **manual acceptance checks** — CI currently covers the Python
  services only; a Node/pnpm job for the web build and lint is not yet wired
  up.
- Streaming chat works end-to-end via the typed API client; backend boundary is
  the only integration point (no direct service calls from components).
- The backend implementation is swappable behind the API client without
  frontend architecture changes.
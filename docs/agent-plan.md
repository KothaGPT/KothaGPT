# AI Agent — Implementation Plan

Goal: a production agent runtime for KothaGPT — a registry of permissioned,
sandboxed tools (browser, search, code execution, files, database), function
calling driven by the aligned model, a planning engine, a resilient agent
loop, memory (short- and long-term), multi-agent orchestration, and a hard
permission + sandbox gate — exposed through the existing `services/api/`
agents/tools surface.

Guiding principles:

- Build on what exists: `services/api/` already exposes `/v1/agents` (CRUD,
  runs, SSE streaming) and `/v1/tools` (list/invoke) behind the pluggable
  `Backend` with mock implementations; the SFT plan (`docs/sft-plan.md`
  WS-9/WS-10) already trains function-calling and tool-use into the model. The
  agent plan makes those real, safe, and observable.
- Safety is architectural: every tool goes through the permission system
  (WS-14) and runs in a sandbox (WS-15). No tool executes outside a sandbox.
  No agent acts outside its permission scope.
- Agents are observable: every step (thought → tool call → result → next step)
  is logged to the run transcript so failures are debuggable and audits
  possible.
- Memory is explicit: short-term (within a run) vs long-term (persistent,
  user-controlled) are separate stores with separate write policies; the model
  never silently persists.
- The loop is bounded and resumable: max steps, timeouts, and interruption
  handling are first-class config; a hung agent is a bug, not a user problem.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| API surface | `/v1/agents` (CRUD + runs + SSE stream) and `/v1/tools` (list + invoke) routers; `Backend` agent/tool methods; mock backend implements all of it | all mock — no real tools, loop, memory, permissions, sandbox |
| Agent service | `services/agents/` exists but is empty | whole workstream missing |
| Function calling | SFT plan WS-9 (function-calling dataset) / WS-10 (tool-use) specify model capability | not trained yet; no runtime tool-call parser/handler |
| Runtime | `docs/runtime-plan.md` engine + registry + auth/rate-limit | agent layer builds on top |
| RAG | `docs/rag-plan.md` retrieval/context/citations | agents may consume retrieval as a tool |
| Data stores | Postgres + Redis + Qdrant in compose | no memory store, no sandbox infra |

---

## Workstreams

### WS-1 — Tool registry

> Foundation slice implemented: local tool registry, permission gate, function-call parsing, and bounded agent loop are available under `services/agents/`. External sandboxing, model dispatch, memory, and orchestration remain pending. (`services/agents/registry.py`)

Goal: one schema-driven catalog of everything an agent may call.

- `ToolSpec {name, description, parameters (JSON schema), permission, sandbox,
  cost, enabled}`; registry stores specs in Postgres, serves `/v1/tools` with
  real data, and validates invocations against the JSON schema before dispatch.
- Version tool specs; a changed signature updates the registry, not the model
  prompt.
- Deliverables: `registry.py`, real `list_tools`/`get_tool`.
- Metric: registry CRUD round-trips; every tool spec validates; invocations
  with bad args are rejected at the boundary with a clear error.

### WS-2 — Function calling (`services/agents/fc.py`)

Goal: turn model output into verifiable tool calls.

- Build on the SFT function-calling capability: a runtime parser that reads the
  model's `<function_call>`/JSON output, validates it against the registry
  (WS-1), dispatches, and feeds the result back into the next model turn
  (2-round tool-use pattern from the SFT plan).
- Handle the failure paths: malformed call → retry/repair once; unknown
  function → clarify; missing args → ask.
- Deliverables: `fc.py`, tool-call verification hooks.
- Metric: call parse + dispatch accuracy ≥ threshold on the held-out
  function-call eval set; zero dispatch of unregistered tools; repair covers ≥
  threshold of malformed calls.

### WS-3 — Browser tool (`services/agents/tools/browser.py`)

Goal: safe web access as a tool.

- Implement a headless-browser tool (navigation, extract text/links, screenshots
  to the sandbox only), allow-listed domains + robots.txt respect (reuse the
  RAG crawler policy), per-call URL budget, and no credential storage.
- Deliverable is a page snapshot (text) the model can reason over; never raw
  scripts.
- Deliverables: `browser.py` + sandbox wiring.
- Metric: tool returns clean text snapshots within latency budget; blocked
  domains never fetched; no script execution or credential capture.

### WS-4 — Search tool (`services/agents/tools/search.py`)

Goal: retrieval as an agent tool, not a bespoke integration.

- Wrap the RAG retriever (WS-7/WS-8 of `docs/rag-plan.md`) as a tool: query →
  top-k chunks + citations; also a web-search mode via the browser/crawler
  path. Return source-marked results the model must cite.
- Deliverables: `search.py` tool.
- Metric: results match the RAG eval quality; answers reference the returned
  sources (citation compliance checked by the WS-11/agent transcript).

### WS-5 — Code execution tool (`services/agents/tools/code.py`)

Goal: run user-approved code, safely.

- Execute Python/JS in the sandbox (WS-15) with no network, no env secrets, a
  strict memory/CPU/time budget, and stdout/stderr capture; interactive REPL
  state per run.
- Language/version pinned per tool spec; result returned as text.
- Deliverables: `code.py`, sandbox runner.
- Metric: code runs within the budget and returns captured output; escapes
  (network, filesystem writes outside scratch) are blocked; timeouts kill the
  process cleanly.

### WS-6 — File tool (`services/agents/tools/files.py`)

Goal: controlled file access confined to the sandbox workspace.

- List/read/write/append within the run's scratch workspace (sandbox-mounted
  dir); enforce path allow-list and size caps; reject path traversal and
  absolute-system writes.
- Deliverables: `files.py` + tests.
- Metric: every file op confined to the workspace; traversal attempts rejected;
  read/write round-trips verified; size caps enforced.

### WS-7 — Database tool (`services/agents/tools/db.py`)

Goal: query user-authorized databases without leaking schema or data.

- Connect to approved DB endpoints (Postgres via compose) under a read-only,
  restricted user; validate statements (no DDL/DML unless explicitly allowed),
  cap row counts and timeouts, and redact PII from returned rows.
- Deliverables: `db.py`, restricted-role fixtures.
- Metric: reads only (no writes unless flag); row/time caps enforced; PII
  redaction verified; unauthorized connections blocked.

### WS-8 — Memory system (`services/agents/memory.py`)

Goal: unified memory API across short- and long-term stores.

- Define the memory interface: `store/recall/forget(key, value, scope)` with
  explicit user consent policy — memory writes are never implicit; a user-visible
  summary of what is stored is always available.
- Backed by Redis (short-term) + Postgres (long-term), namespaced per
  user/agent.
- Deliverables: `memory.py` + consent/audit hooks.
- Metric: store/recall round-trips in both stores; scoping and consent enforced;
  audit log records every write.

### WS-9 — Short-term memory (`services/agents/memory_short.py`)

Goal: within-run working memory so the loop doesn't lose thread.

- Run-scoped context: recent steps, tool results, intermediate facts with a
  rolling window + relevance pruning; survives multi-step loops, expires with
  the run.
- Deliverables: `memory_short.py` (Redis-backed, TTL).
- Metric: facts survive N steps within a run; TTL expiry is honored; no leakage
  across runs/users.

### WS-10 — Long-term memory (`services/agents/memory_long.py`)

Goal: persistent, user-controlled knowledge across sessions.

- Postgres-backed persistent store: user facts/preferences with timestamps,
  sources, and explicit consent; supports forget/expire and per-key ACLs; only
  the agent's authorized scope can write.
- Deliverables: `memory_long.py` + consent API.
- Metric: persistence across runs verified; forget removes the key and its
  audit record; cross-user isolation tested.

### WS-11 — Planning engine (`services/agents/planner.py`)

Goal: decompose tasks into executable, checkable steps.

- Given a goal, produce a plan (ordered steps with dependencies, required
  tools, expected outputs) via the aligned model; re-plan when a step fails or
  the environment changes.
- Keep plans bounded (max steps, cost estimate) and expose the plan in the
  transcript before execution.
- Deliverables: `planner.py` + plan schema.
- Metric: plans cover the eval task set with ≥ threshold success on first try;
  re-plan triggers on injected failures; plan length within budget.

### WS-12 — Agent loop (`services/agents/loop.py`)

Goal: a robust observe-think-act cycle.

- Implement the loop: `observe (transcript+memory) → think (model) → act
  (tool call or answer) → observe (result)` with max steps, timeouts, stop
  conditions, and interruption/resume; streams events to `/v1/agents/{id}/runs`
  via the existing SSE path.
- Guard rails: loop progress monitor (stall/cycle detection), budget caps,
  and a final answer always includes the transcript summary.
- Deliverables: `loop.py`, run transcript store.
- Metric: eval tasks complete within budget; stall/cycle detection fires on
  synthetic loops; streams match the SSE schema; resume reproduces state.

### WS-13 — Multi-agent orchestration (`services/agents/orchestrator.py`)

Goal: coordinate specialist agents without chaos.

- Orchestrator pattern: a supervisor agent delegates subtasks to registered
  specialist agents (browser/search/code/data), collects results, and merges
  into the final answer; per-subtask permission propagation from the caller.
- Bounded fan-out, per-subtask budgets, and no agent-talking-to-agent loops
  without the supervisor's oversight.
- Deliverables: `orchestrator.py` + delegation eval.
- Metric: delegation tasks beat single-agent baseline on the eval set; subtask
  budgets enforced; permission scope never widens down the tree.

### WS-14 — Agent permission system (`services/agents/permissions.py`)

Goal: nothing executes outside its granted scope.

- Per-agent/per-user permission matrix (tool, resource, budget, sandbox
  profile); enforced at tool dispatch (WS-2), memory writes (WS-8), and
  orchestration (WS-13); all denials are logged and streamed to the transcript.
- Admin approval flows for high-risk tools (code, db) — allow once, allow
  session, deny.
- Deliverables: `permissions.py`, admin/approval API.
- Metric: denied calls never execute (e2e test); approval flow round-trips;
  audit log captures every decision.

### WS-15 — Agent sandbox (`services/agents/sandbox.py`)

Goal: defense-in-depth isolation for every tool.

- Sandbox per run/agent: OS-level isolation (separate user/container/network
  namespace via the compose `gpu`/agent profile), scratch workspace mount,
  no network except allow-listed egress, CPU/mem/time limits, and filesystem
  confinement (WS-5/WS-6 depend on this).
- Teardown guarantees: no sandbox survives past its run timeout; artifacts
  harvested to the workspace then destroyed.
- Deliverables: `sandbox.py`, sandbox infra in compose, escape tests.
- Metric: escape tests (network, file, process) are all blocked; teardown
  leaves no processes/containers behind; resource limits enforced.

---

## Sequencing & dependencies

```
WS-1 tool registry ──> WS-2 function calling ──> WS-12 agent loop
                               │                      │
WS-3..7 tools ─────────────────┘                      ├──> WS-11 planner
WS-8 memory ──> WS-9 short-term ──> WS-10 long-term ───┤
WS-14 permissions ──> WS-15 sandbox ───────────────────┤ (gates all tools)
WS-13 orchestration ───────────────────────────────────┘
```

WS-1 → WS-2 → WS-12 is the critical path (registry → function calling →
loop). WS-3..WS-7 tools plug into WS-2 and are gated by WS-14/WS-15 — the
permission + sandbox gate ships *before* any real tool is enabled. WS-8..WS-10
memory and WS-11 planning enrich the loop in parallel. WS-13 (multi-agent)
sits on top of WS-12.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Tool registry | WS-1 |
| Function calling | WS-2 |
| Browser tool | WS-3 |
| Search tool | WS-4 |
| Code execution tool | WS-5 |
| File tool | WS-6 |
| Database tool | WS-7 |
| Memory system | WS-8 |
| Short-term memory | WS-9 |
| Long-term memory | WS-10 |
| Planning engine | WS-11 |
| Agent loop | WS-12 |
| Multi-agent orchestration | WS-13 |
| Agent permission system | WS-14 |
| Agent sandbox | WS-15 |

## Tests

```bash
python -m pytest tests/test_tool_registry.py      # WS-1
python -m pytest tests/test_function_calling.py   # WS-2
python -m pytest tests/test_browser_tool.py       # WS-3
python -m pytest tests/test_search_tool.py        # WS-4
python -m pytest tests/test_code_tool.py          # WS-5
python -m pytest tests/test_file_tool.py          # WS-6
python -m pytest tests/test_db_tool.py            # WS-7
python -m pytest tests/test_memory.py             # WS-8
python -m pytest tests/test_memory_short.py       # WS-9
python -m pytest tests/test_memory_long.py        # WS-10
python -m pytest tests/test_planner.py            # WS-11
python -m pytest tests/test_agent_loop.py         # WS-12
python -m pytest tests/test_orchestrator.py       # WS-13
python -m pytest tests/test_permissions.py        # WS-14
python -m pytest tests/test_sandbox.py            # WS-15
make eval-agent                                    # end-to-end agent suite
```

Never commit sandbox state, memory contents, or agent transcripts;
`services/agents/` runtime data is git-ignored. Agents consume the aligned
model (`docs/sft-plan.md` function-calling) + RAG retrieval (`docs/rag-plan.md`)
and sit on the runtime registry (`docs/runtime-plan.md`); the mock backend
stays the CI/dev fallback.

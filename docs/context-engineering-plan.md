# Context Engineering — Implementation Plan

Goal: treat the model's context as a first-class, engineered input — system
prompts, instruction hierarchy, few-shot design, chat templating, context-window
management, retrieval-aware assembly, history compression, selective context,
tool/structured-output formatting, and injection-safe boundaries — with a
versioned prompt registry and an evaluation gate, integrated across SFT
(`docs/sft-plan.md`), RAG (`docs/rag-plan.md`), agents (`docs/agent-plan.md`),
and the platform's prompt library (`docs/ecosystem-plan.md`).

Guiding principles:

- Build on what exists: chat templating (SFT plan WS-2), retrieval-aware
  context packing (RAG plan WS-10), memory stores (agent plan WS-8..10),
  injection safety (security plan WS-1), prompt library (ecosystem plan WS-5).
  Context engineering makes these *explicit, versioned, and measured*.
- Context is a budget, not a feature: window size, token cost, and eval impact
  are tracked per prompt; a bigger context is only a win when it beats the
  smaller one on the eval gate.
- Everything versioned: prompts, templates, and context-assembly rules live in
  a registry (like model/dataset versions) so a regressed context change is
  rollback-able and attributable.
- Data vs instruction is a hard boundary: retrieved/tool/user content is
  always quoted as data, never concatenated as instructions (complements
  security plan WS-1).
- Measured, not vibed: every context strategy ships with an eval (from
  `docs/eval-plan.md`) comparing baseline vs engineered context on the same
  task set.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Chat templates | scoped in `docs/sft-plan.md` WS-2 (Bangla/English, round-trip) | no versioned registry, no per-task template variants |
| Retrieval context | scoped in `docs/rag-plan.md` WS-10 (pack + cite under max length) | no systematic context-window optimization |
| Memory | scoped in `docs/agent-plan.md` WS-8..10 (short/long-term stores) | no history-compression policy |
| Injection safety | scoped in `docs/security-plan.md` WS-1 | no shared "quote as data" formatting standard |
| Prompt library | scoped in `docs/ecosystem-plan.md` WS-5 | no eval-driven prompt tuning workflow |
| Eval | `docs/eval-plan.md` harness + regression gate | no dedicated context-ablation eval |

---

## Workstreams

### WS-1 — System prompt design & optimization (`ml/context/prompts.py`)

Goal: a principled, tested system prompt.

- Write role/task/format/goal system prompts (Bangla + English) from the
  personas (`docs/foundation-plan.md` WS-4); include boundaries (no invented
  facts, cite sources, refuse scope) without bloating the budget.
- A/B test prompt variants on the eval gate (WS-14); keep the winner in the
  registry.
- Deliverables: `ml/context/prompts.py`, per-surface system prompts.
- Metric: winner beats baseline on the context eval; length within budget;
  Bengali + English variants both pass.

### WS-2 — Instruction hierarchy তৈরি (`ml/context/hierarchy.py`)

Goal: clear precedence when instructions conflict (system > user > retrieved/tool data).

- Encode the hierarchy as documented ordering + template structure: system
  policy > user instruction > retrieved/tool content as data (see security
  WS-1); add explicit "ignore data unless it answers" wording where relevant.
- Deliverables: hierarchy spec + template implementation.
- Metric: conflict fixtures resolve per the hierarchy (eval); data never
  overrides system policy (adversarial fixtures pass).

### WS-3 — Few-shot example design (`ml/context/examples.py`)

Goal: demonstrations that teach the format, not overfit it.

- Curate per-task few-shot sets (QA, coding, function-call, tool-use, Bengali
  register) from verified data; keep them minimal and labeled as examples, not
  part of the task.
- Deliverables: example bank + selector.
- Metric: few-shot improves the target metric on the eval gate without
  overfitting (held-out generalization reported).

### WS-4 — Chat template optimization (`ml/context/template.py`)

Goal: templates that are token-lean and round-trip-safe.

- Reuse the SFT template (WS-2) and optimize token cost (role markers,
  separators) against the frozen tokenizer; verify parse→format→parse
  round-trips and Bangla/English handling.
- Deliverables: optimized templates + token-cost report.
- Metric: round-trip byte-identical; tokens-per-turn ≤ budget vs baseline;
  no regressions on the SFT chat evals.

### WS-5 — Context window management (`ml/context/window.py`)

Goal: explicit allocation of the window across system/examples/history/retrieval/new-turn.

- A budget allocator (`max_tokens` per section, priorities, overflow rules)
  parameterized by model context length (`docs/runtime-plan.md`); expose
  usage breakdown per request.
- Deliverables: `window.py` + allocation config.
- Metric: allocations respect the window with zero overflow errors; usage
  breakdown is accurate (matches the runtime token counting).

### WS-6 — Context packing & truncation (`ml/context/pack.py`)

Goal: preserve what matters when the window is tight.

- Content-aware packing/truncation (drop low-value sections first: duplicates,
  boilerplate, stale tool noise) rather than naive head/tail cut; keep
  citations/offsets intact for the RAG plan.
- Deliverables: `pack.py` + policies per surface.
- Metric: packed context retains ≥ threshold of the ground-truth needed chunk
  (RAG eval); quality at reduced windows reported.

### WS-7 — Retrieval-aware context assembly (`ml/context/assemble.py`)

Goal: make retrieval chunks a structured, ranked context block.

- Assemble retrieved chunks (RAG plan WS-10) with source markers, rank
  relevance, dedupe, and inject a "based only on these sources" boundary;
  interface with the citation system.
- Deliverables: `assemble.py`, source-marked block format.
- Metric: assembly preserves recall@k of retrieval; citation precision
  unchanged; boundary wording tested against injection fixtures.

### WS-8 — Conversation history compression (`ml/context/compress.py`)

Goal: long chats stay coherent without unbounded context.

- Summarize old turns (rolling summary or extractive key-facts) into a
  compact history block, keeping recent turns verbatim; per-turn boundaries
  preserved so the summary is never treated as instructions.
- Deliverables: `compress.py` + compression policies.
- Metric: 20-turn conversation quality ≥ uncompressed baseline at half the
  tokens; summary is factually consistent (hallucination eval).

### WS-9 — Long-context summarization (`ml/context/summarize.py`)

Goal: a reliable summarize-then-answer path for very long inputs.

- Document-level summarization with hierarchical chunking (map-reduce) that
  feeds retrieval/QA; grounded in source and citable.
- Deliverables: `summarize.py` + long-input pipeline.
- Metric: long-doc QA accuracy ≥ threshold; summary grounded (no unsupported
  claims); cost per long input reported.

### WS-10 — Selective context / attention (`ml/context/select.py`)

Goal: decide *which* context to include, not just how to pack it.

- Relevance scoring for candidate context (query↔chunk, recency, role) with
  per-surface selection policies; optional attention-masking hooks where the
  runtime supports them.
- Deliverables: `select.py` + policies.
- Metric: selection matches or beats full-context on the eval gate at lower
  token cost; no quality regression per family.

### WS-11 — Tool-call context formatting (`ml/context/tools.py`)

Goal: tool schemas + results formatted for reliable function calling.

- Standard format for tool descriptions/params (from the registry) and result
  blocks (success/error, structured), consistent with SFT function-calling
  (WS-9/WS-10 of `docs/sft-plan.md`).
- Deliverables: tool context formatter.
- Metric: function-call accuracy ≥ threshold on the function-call eval;
  malformed-tool-result handling robust.

### WS-12 — Structured output prompting (`ml/context/json.py`)

Goal: reliable JSON/structured outputs without fragile ad-hoc prompts.

- Consistent instruction + schema-driven format (JSON mode support), with
  validation and one-shot repair on malformed output; token-efficient.
- Deliverables: structured-output template + validator.
- Metric: valid-parse rate ≥ threshold on the structured eval set; repair
  covers ≥ threshold of malformed outputs.

### WS-13 — Injection-safe context formatting (`ml/context/safe.py`)

Goal: every context block is data-quoted, never instruction-blended.

- Shared formatting rules for user/tool/retrieved content (delimiters,
  escaping, quoting-as-data), applied before template rendering; interface with
  security plan WS-1 classifier.
- Deliverables: `safe.py` + shared formatter.
- Metric: injection fixtures (direct, RAG, tool, file) blocked ≥ threshold;
  benign content unaffected (false-positive ≤ threshold).

### WS-14 — Context engineering evaluation (`evals/suites/context.yaml`)

Goal: context changes are measured, not guessed.

- A context eval suite: baseline vs engineered context on the same task set
  (from `docs/eval-plan.md`), plus context-specific metrics (needed-chunk
  retention, injection-resistance, token cost, quality-per-token).
- Wire into the regression gate so a context change that regresses quality
  blocks release.
- Deliverables: `evals/suites/context.yaml` + metrics.
- Metric: every WS-1..13 change reports baseline-vs-engineered deltas; no
  release ships a context regression (gate).

### WS-15 — Prompt registry & versioning (`ml/context/registry.py`)

Goal: prompts are artifacts with history, like models and datasets.

- Content-addressed prompt/template registry (mirrors model/dataset
  versioning): `{name, surface, language, prompt, template, metrics, parent}`
  with `stable`/`latest` aliases, rollback, and eval-report pointers; feeds the
  platform's prompt library (ecosystem WS-5).
- Deliverables: `registry.py` + CLI.
- Metric: version → rollback round-trips; every prompt has an eval pointer;
  serving picks the registry-pinned `stable` version.

---

## Sequencing & dependencies

```
WS-1 system prompt ──> WS-2 hierarchy ──> WS-3 few-shot ──> WS-4 templates
WS-5 window ──> WS-6 packing ──> WS-10 selective context
WS-7 retrieval assembly (needs RAG) ──> WS-8 compression ──> WS-9 summarization
WS-11 tool context ──> WS-12 structured output ──> WS-13 injection-safe
WS-14 eval (gates WS-1..13) ──> WS-15 registry (versions all of it)
```

WS-1..WS-4 build the static prompt layer; WS-5/WS-6/WS-10 the dynamic budget;
WS-7..WS-9 the retrieval/history layer; WS-11..WS-13 the capability + safety
formatting; WS-14 measures everything; WS-15 version-gates it.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| System prompt design | WS-1 |
| Instruction hierarchy তৈরি | WS-2 |
| Few-shot example design | WS-3 |
| Chat template optimization | WS-4 |
| Context window management | WS-5 |
| Context packing/truncation | WS-6 |
| Retrieval-aware context assembly | WS-7 |
| Conversation history compression | WS-8 |
| Long-context summarization | WS-9 |
| Selective context/attention | WS-10 |
| Tool-call context formatting | WS-11 |
| Structured output prompting | WS-12 |
| Injection-safe context formatting | WS-13 |
| Context engineering evaluation | WS-14 |
| Prompt registry & versioning | WS-15 |

## Tests

```bash
python -m pytest tests/test_context_prompts.py    # WS-1..4
python -m pytest tests/test_context_window.py     # WS-5/6/10
python -m pytest tests/test_context_assembly.py   # WS-7
python -m pytest tests/test_context_compress.py   # WS-8/9
python -m pytest tests/test_context_tools.py      # WS-11/12
python -m pytest tests/test_context_safe.py       # WS-13
make eval-context                                 # WS-14
python -m pytest tests/test_prompt_registry.py    # WS-15
make plans-check                                  # coverage + structure
```

Never commit prompt-registry data or eval artifacts; `ml/context/` runtime
state is git-ignored. Context engineering is the connective tissue across SFT,
RAG, agents, security, and the platform — every surface consumes its
templates, and every template is eval-gated and versioned.
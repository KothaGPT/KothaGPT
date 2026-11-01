# Production Model Family — Implementation Plan

Goal: operate the right model for each problem class — Nano, Small, Medium, Large — with a
MoE research track for frontier efficiency, strong reasoning and coding stacks,
advanced multimodal pipelines, and stronger evaluation and adversarial coverage.

Guiding principles:

- **Right-sized models**: every problem class has a dedicated tier with explicit
  cost/performance targets; routing sends requests to the cheapest tier that
  meets the quality floor.
- **MoE as the frontier path**: dense models scale predictably; Mixture-of-Experts
  delivers frontier quality at a fraction of the FLOPs. Research first, production
  second.
- **Specialist stacks**: reasoning and coding get dedicated training recipes,
  evaluation suites, and model variants — not just config toggles.
- **Multimodal is first-class**: vision, audio, and document understanding ship
  as parallel encoder paths, not afterthoughts.
- **Eval gates everything**: no model tier ships without passing its
  tier-specific eval gate; adversarial coverage is the release veto.
- Build on what exists: `ml/configs/*.yaml`, `ml/models/`, `ml/inference/`,
  `ml/trainer/`, `ml/sft/`, `evals/`, and the `ModelRegistry` are the
  foundation. Every new tier plugs into the existing registry, loader, and API.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Model tiering | `small.yaml` (768/12L), `long.yaml` (16k ctx), `smoke.yaml` (192/2/4), SFT configs (128/4L) | no formal tiering taxonomy; no Nano/Medium/Large configs |
| Registry | `ModelRegistry` with `kothagpt`, `kothagpt-small`, `kothagpt-embed`, `kothagpt-rerank` | no tier fields, no MoE entries, no multimodal entries |
| Architecture | decoder-only transformer (SwiGLU, RoPE, RMSNorm) | no MoE layer, no multimodal encoder, no reasoning-specific head |
| Trainer | `ml/trainer/` loop, checkpoint, scheduler, monitor, DDP/FSDP | no tier-aware training, no MoE training recipe |
| SFT | `ml/sft/` with bn/en/multilingual/code variants | no reasoning/coding-specialist SFT pipeline, no tier-specific SFT |
| Eval | `evals/` with general, bangla, english, coding, reasoning, math, RAG, agent, hallucination, safety, latency, token-efficiency | no tier-specific eval gates, no adversarial suite, no MoE eval |
| Inference | `ml/inference/` engine, KV cache, batcher, quant, loader, registry, version | no tier-aware router, no MoE inference engine, no multimodal decoder |

---

## Workstreams

### WS-1 — Kotha Nano / Small / Medium / Large tiering

Goal: define, configure, and train four explicit model tiers with
documented cost/performance profiles.

#### Tier definitions

| Tier | Params | Hidden | Layers | Heads | Context | Target Use | Cost Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Nano** | ~0.3B | 256 | 8 | 4 | 2048 | Edge/mobile, low-latency chat, autocomplete | <$0.001/1k tokens |
| **Small** | ~1.5B | 768 | 12 | 12 | 4096 | General chat, coding assist, RAG | <$0.005/1k tokens |
| **Medium** | ~7B | 2048 | 24 | 16 | 8192 | Reasoning, long-context, agent workflows | <$0.02/1k tokens |
| **Large** | ~14B+ | 4096 | 32 | 32 | 16384 | Frontier reasoning, multimodal, research | <$0.08/1k tokens |

#### Deliverables

- `ml/configs/nano.yaml` — Nano tier config (256/8/4, ctx 2048, minimal compute)
- `ml/configs/small.yaml` — updated Small tier config (768/12/12, ctx 4096, current default)
- `ml/configs/medium.yaml` — Medium tier config (2048/24/16, ctx 8192)
- `ml/configs/large.yaml` — Large tier config (4096/32/32, ctx 16384)
- `ml/models/tier.py` — `ModelTier` enum + `TIER_CONFIGS` dict mapping tier → config path
- `ml/models/config.py` — extend `ModelConfig` with `tier`, `num_experts` (for MoE), `mm_hidden` (multimodal)
- `ml/inference/registry.py` — register all four tiers with metadata (params, context, quant levels, device profile)

#### Config specs

**Nano** (`ml/configs/nano.yaml`):
- vocab_size: 50000 (frozen tokenizer), hidden_size: 256, num_layers: 8, num_heads: 4
- intermediate_size: 1024 (4× hidden), max_position_embeddings: 2048
- rms_norm_eps: 1e-5, rope_theta: 10000, tie_word_embeddings: true
- training: batch 4, grad_accum 8, lr 3e-4, max_steps 500, mixed_precision bf16
- context_window: 2048, device_profile: cpu, quant_level: int8

**Small** (`ml/configs/small.yaml`): update existing — hidden 768, layers 12, heads 12, ctx 4096

**Medium** (`ml/configs/medium.yaml`):
- vocab_size: 50000, hidden_size: 2048, num_layers: 24, num_heads: 16
- intermediate_size: 8192, max_position_embeddings: 8192
- rope_theta: 500000, rope_scaling: linear, rope_scaling_factor: 2.0
- training: batch 2, grad_accum 16, lr 1.5e-4, max_steps 2000, mixed_precision bf16
- context_window: 8192, device_profile: gpu, quant_level: int8

**Large** (`ml/configs/large.yaml`):
- vocab_size: 50000, hidden_size: 4096, num_layers: 32, num_heads: 32
- intermediate_size: 16384, max_position_embeddings: 16384
- rope_theta: 1000000, rope_scaling: linear, rope_scaling_factor: 4.0
- training: batch 1, grad_accum 32, lr 1e-4, max_steps 4000, mixed_precision bf16
- context_window: 16384, device_profile: gpu, quant_level: int4

#### Metric: each tier's config loads and validates; `ModelTier` resolves config path to digest; registry lists all four tiers with accurate metadata.

### WS-2 — MoE research track for frontier efficiency

Goal: research and prototype Mixture-of-Experts architecture for
frontier-quality models at reduced compute cost.

#### Research phases

**Phase A — Architecture research**
- Implement `ml/models/moe/` package:
  - `SparseMoELayer`: top-k gating (k=2), load-balanced expert routing
  - `ExpertFFN`: SwiGLU expert with shared expert fallback
  - `MoEBlock`: transformer block with MoE FFN replacing dense FFN
  - `Router`: softmax gating with capacity factor, expert-choice routing
- Compare dense vs MoE at fixed parameter budget (e.g., 7B dense vs 7B MoE with 8 experts)
- Deliverables: `moe/layer.py`, `moe/router.py`, `moe/block.py`, `moe/decision.md`

**Phase B — Training recipe**
- Auxiliary loss for load balancing (expert utilization)
- Capacity factor tuning (prevent expert collapse)
- Training stability: lower LR for MoE layers, gradual capacity warmup
- Deliverables: `moe/train.py`, `moe/loss.py`, training stability report

**Phase C — Inference optimization**
- Static routing table (pre-compute top-2 experts per token bucket)
- Batching experts across requests (expert parallelism)
- Quantization-friendly MoE (int8 experts, fp16 gating)
- Deliverables: `ml/inference/moe_engine.py`, `ml/inference/expert_parallel.py`

**Phase D — Frontier model**
- `ml/configs/moe.yaml` — MoE variant (e.g., 7B total params, 8 experts, 1B active per token)
- Train or fine-tune from Medium checkpoint
- Evaluate against dense Medium and Large baselines
- Deliverables: `moe.yaml`, MoE checkpoint, evaluation report

#### Metric: MoE model achieves ≥ 90% of dense model quality at ≤ 50% of FLOPs; expert load balance > 0.8; inference latency within 1.2× dense baseline.

### WS-3 — Strong reasoning and coding stacks

Goal: dedicated reasoning and coding model variants with
specialized training recipes, evaluation suites, and routing.

#### Reasoning stack

- **Training**: `ml/configs/sft-reasoning.yaml` — SFT from Medium/Large base with
  reasoning-heavy mixture (≥ 60% reasoning: math, logic, multi-hop, commonsense)
  plus verifiable reward signal for DPO preference tuning
- **Architecture**: add a `reasoning_head` optional module — a supervised
  decoder head on top of the LM head that predicts solution steps;
  used during training for auxiliary supervision, not during inference
- **Inference-time**: optional chain-of-thought generation with
  verifier loop (self-consistency, verifier-based filtering)
- **Config**: extended context (8192+) for multi-step reasoning,
  temperature 0.0–0.3 for factual accuracy
- **Eval**: `evals/suites/reasoning.yaml` with `reasoning_math`,
  `reasoning_logic`, `reasoning_multihop` task families;
  pass@k, verifier agreement, self-consistency rate metrics

#### Coding stack

- **Training**: `ml/configs/sft-coding.yaml` — SFT from Medium/Large base with
  coding-heavy mixture (≥ 60% code: function synthesis, bug-fix,
  explanation, Bangla-comment code) plus execution-based reward
- **Architecture**: code-specific tokenizer extensions (code tokens,
  special markers for `<code>`, `</code>`, `<function>`, `</function>`)
  registered in `ml/tokenizer` as `coding` special tokens
- **Inference-time**: execution sandbox integration (`services/agents/tools/code.py`)
  for pass@k evaluation; structured output mode for JSON function calls
- **Config**: extended context (16384) for multi-file context,
  temperature 0.0–0.2 for code accuracy
- **Eval**: `evals/suites/coding.yaml` with `coding_function_synthesis`,
  `coding_bug_fix`, `coding_explanation` task families;
  pass@k on unit-test execution, syntax validity, Bangla-comment handling

#### Deliverables

- `ml/configs/sft-reasoning.yaml`, `ml/configs/sft-coding.yaml`
- `ml/sft/reasoning.py`, `ml/sft/coding.py` — specialist dataset loaders
- `ml/models/reasoning_head.py` — optional reasoning auxiliary head
- `ml/tokenizer/coding.py` — coding-specific token extensions
- `evals/suites/reasoning.yaml`, `evals/suites/coding.yaml`
- `services/agents/tools/code.py` — already exists; integrate with eval runner

#### Metric: reasoning model achieves ≥ 75% pass@1 on math/logic benchmarks; coding model achieves ≥ 60% pass@k on unit-test execution; zero regression on general benchmarks.

### WS-4 — Advanced multimodal pipelines

Goal: vision, audio, and document understanding as parallel
encoder paths with unified decoder, enabling rich multimodal inputs.

#### Architecture

- **Vision encoder**: frozen ViT (or CLIP-style) encoder projecting
  image patches into the model's hidden space; `mm_hidden` config field
  controls the projection dimension
- **Audio encoder**: frozen audio encoder (Whisper-style or HuBERT)
  projecting audio features into hidden space
- **Document encoder**: OCR + layout-aware encoder for document understanding
- **Unified decoder**: existing KothaGPT decoder with multimodal-aware
  attention layers that fuse cross-modal context

#### Implementation phases

**Phase A — Vision**
- `ml/models/multimodal/vision.py`: `VisionEncoder` + `VisionProjector`
  (MLP or Perceiver resolver projecting to `hidden_size`)
- Image preprocessing pipeline: resize, patch extraction, positional encoding
- Deliverables: `vision.py`, `ml/configs/mm-vision.yaml`

**Phase B — Audio**
- `ml/models/multimodal/audio.py`: `AudioEncoder` + `AudioProjector`
- Audio preprocessing: spectral features or raw waveform → token-like embeddings
- Deliverables: `audio.py`, `ml/configs/mm-audio.yaml`

**Phase C — Document**
- `ml/models/multimodal/document.py`: `DocumentEncoder` combining
  OCR text + layout features + page structure
- Deliverables: `document.py`, `ml/configs/mm-document.yaml`

**Phase D — Unified model**
- `ml/models/multimodal/unified.py`: `KothaMultimodal` extending `KothaGPT`
  with multimodal encoders, cross-modal attention, and unified generation
- `ml/configs/mm-unified.yaml`: combined multimodal config
- Deliverables: `unified.py`, `mm-unified.yaml`, multimodal eval suite

#### Deliverables

- `ml/models/multimodal/{vision,audio,document,unified,__init__}.py`
- `ml/configs/mm-*.yaml` (vision, audio, document, unified)
- `ml/sft/mm_dataset.py` — multimodal instruction dataset loader
- `evals/suites/multimodal.yaml` — multimodal eval suite

#### Metric: vision encoder achieves ≥ 80% of CLIP image-text similarity baseline; audio encoder achieves ≥ 0.7 WER on Bangla speech; document encoder achieves ≥ 70% layout-aware extraction accuracy; unified model handles at least 2 modalities in a single request.

### WS-5 — Model routing system

Goal: intelligent routing that sends each request to the right
model tier based on task complexity, achieving optimal cost/performance balance.

#### Router architecture

- `ml/inference/router.py`: `ModelRouter` class with:
  - `route(request) → tier_id`: classify request by complexity
  - Complexity signals: prompt length, expected tokens, task type hint,
    user tier/priority, latency SLA
  - Fallback chain: Nano → Small → Medium → Large (auto-escalation
    when quality floor not met)
  - Circuit breaker: if a tier is overloaded or failing, route to next
- `services/api/router.py`: HTTP-level routing integration;
  `X-Kotha-Tier` header for explicit tier selection; auto-tier header
  for router-controlled selection

#### Routing rules

| Task class | Default tier | Escalation |
| --- | --- | --- |
| Autocomplete, short chat (< 100 tokens) | Nano | Small |
| General conversation, RAG, coding assist | Small | Medium |
| Reasoning, long-context, agent workflows | Medium | Large |
| Multimodal, frontier research, complex analysis | Large | — |
| Explicit user tier preference | User-specified | — |

#### Deliverables

- `ml/inference/router.py` — `ModelRouter` with complexity classifier
- `services/api/router.py` — HTTP routing integration
- `ml/inference/router_config.yaml` — routing rules and tier thresholds
- `evals/suites/routing.yaml` — routing eval (cost/performance balance)

#### Metric: routing improves cost/performance balance by ≥ 30% vs.
always-large baseline; escalation rate < 10%; p95 latency within SLA
for each tier.

### WS-6 — Stronger evaluation and adversarial coverage

Goal: comprehensive evaluation suite with adversarial testing as a
release veto.

#### Evaluation extensions

**Tier-specific eval gates** (`evals/suites/tier-*.yaml`):
- Each tier has its own eval suite calibrated to its capability level
- Nano: basic language understanding, short-context quality
- Small: general knowledge, coding assist, RAG, conversation
- Medium: reasoning, long-context, agent workflows, coding
- Large: frontier reasoning, multimodal, complex analysis
- A tier cannot graduate to production until it passes its tier gate

**Adversarial suite** (`evals/suites/adversarial.yaml`):
- Prompt injection: 500+ injection patterns from `services/security/injection.py`
  blocklists + red-team generated patterns
- Jailbreak: systematic attempts to bypass safety guardrails
- Hallucination probes: factual-consistency tests with known-false premises
- Bias and fairness: demographic bias probes, stereotype tests
- Robustness: typos, code-switching (bn/en mixed), truncation, adversarial suffixes
- Adversarial multiturn: multi-turn attacks designed to bypass context
  filters and safety systems

**Red-team testing** (`evals/suites/redteam.yaml`):
- Structured red-team drills with documented attack trees
- Per-attack success rate tracking
- Regression gate: any new model must not increase attack success rate
  above baseline

**Regression gate** (`evals/regression.py` — extend):
- Baseline store per tier per version (`evals/registry.py`)
- Statistical comparison (mean-CI, hard gates on safety/hallucination)
- Auto-fail on any safety or hallucination regression
- Per-tier gates: Nano gate is lenient on reasoning but strict on latency;
  Large gate is strict on everything

#### Deliverables

- `evals/suites/tier-nano.yaml`, `evals/suites/tier-small.yaml`,
  `evals/suites/tier-medium.yaml`, `evals/suites/tier-large.yaml`
- `evals/suites/adversarial.yaml`, `evals/suites/redteam.yaml`
- `evals/adversarial.py` — adversarial test generator and runner
- `evals/redteam.py` — red-team drill orchestrator
- Extended `evals/regression.py` with tier-aware gates
- `evals/suites/routing.yaml` — routing efficiency eval

#### Metric: every tier passes its tier gate before production; adversarial
success rate < 5%; red-team drill failure rate < 2%; zero safety or
hallucination regressions across all tiers.

---

## Sequencing & dependencies

```
WS-1 tier configs ──> WS-5 routing ──> WS-6 eval gates
   │                                        │
WS-2 MoE research ──> WS-4 multimodal ──> WS-6 eval
   │                                        │
WS-3 reasoning/coding ──> WS-6 eval ──────┘
```

WS-1 is the foundation — tier configs enable routing, eval, and registry.
WS-2 and WS-3 are parallel research tracks feeding the frontier model.
WS-4 (multimodal) depends on WS-1's extended `ModelConfig` (mm_hidden).
WS-5 (routing) depends on WS-1 (tiers) and WS-6 (eval quality floors).
WS-6 (eval) gates all other workstreams and is the last to ship each tier.

### Phase ordering

- **Phase 1 (Weeks 1–3)**: WS-1 tier configs + registry + eval scaffolding
- **Phase 2 (Weeks 2–5)**: WS-3 reasoning/coding stacks (parallel with Phase 1)
- **Phase 3 (Weeks 3–7)**: WS-5 routing + WS-6 adversarial coverage
- **Phase 4 (Weeks 4–8)**: WS-2 MoE research + WS-4 multimodal
- **Phase 5 (Weeks 6–10)**: Integration: routing + tier gates + production rollout
- **Phase 6 (Ongoing)**: MoE frontier model, multimodal expansion, red-team drills

---

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Kotha Nano / Small / Medium / Large tiering | WS-1 |
| MoE research track for frontier efficiency | WS-2 |
| Strong reasoning and coding stacks | WS-3 |
| Advanced multimodal pipelines | WS-4 |
| Model routing improves cost/performance balance | WS-5 |
| Stronger evaluation and adversarial coverage | WS-6 |
| Quality improvement tied to real task outcomes | WS-6 |
| Reliability and safety stable under multi-model deployment (routing) | WS-5 |
| Reliability and safety stable under multi-model deployment (adversarial) | WS-6 |

---

## Success metrics

| Metric | Target |
| --- | --- |
| Model routing improves cost/performance balance | ≥ 30% improvement vs always-large |
| Quality improvement tied to real task outcomes | Each tier passes its tier-specific eval gate |
| Reliability and safety stable under multi-model deployment | Zero safety/hallucination regressions; adversarial success rate < 5% |
| Nano tier latency | p95 < 100ms on CPU |
| Small tier latency | p95 < 300ms on CPU, < 100ms on GPU |
| Medium tier latency | p95 < 1s on GPU |
| Large tier latency | p95 < 2s on GPU |
| MoE efficiency | ≥ 90% dense quality at ≤ 50% FLOPs |
| Reasoning pass@k | ≥ 75% on math/logic benchmarks |
| Coding pass@k | ≥ 60% on unit-test execution |
| Adversarial success rate | < 5% |
| Red-team drill failure rate | < 2% |
| Expert load balance (MoE) | > 0.8 |
| Multimodal quality | ≥ 80% of baseline encoders |

---

## Tests

```bash
# Tier configs
python -m pytest tests/test_model_tier.py        # WS-1
python -m pytest tests/test_tier_configs.py       # WS-1
make config-validate                                # validate all tier configs

# MoE
python -m pytest tests/test_moe_layer.py          # WS-2
python -m pytest tests/test_moe_router.py         # WS-2
make moe-smoke                                    # WS-2 training smoke

# Reasoning / Coding
python -m pytest tests/test_reasoning_head.py     # WS-3
python -m pytest tests/test_coding_tokens.py      # WS-3
make eval-reasoning                               # WS-3 eval
make eval-coding                                  # WS-3 eval

# Multimodal
python -m pytest tests/test_vision_encoder.py     # WS-4
python -m pytest tests/test_audio_encoder.py      # WS-4
make eval-multimodal                              # WS-4 eval

# Routing
python -m pytest tests/test_model_router.py       # WS-5
make eval-routing                                 # WS-5 eval

# Evaluation & Adversarial
make eval-adversarial                             # WS-6
make eval-redteam                                 # WS-6
make eval-tier-gate                               # WS-6 tier gates
make eval-regress                                 # WS-6 regression

# Full suite
make eval-all && make eval-adversarial && make eval-redteam
```

Never commit model weights or training artifacts; `ml/pretrain/artifacts/`,
`ml/sft/artifacts/`, `ml/inference/artifacts/`, and `evals/human/judgments/`
are git-ignored. Every tier's checkpoint and eval report is versioned in the
registry (`ml/inference/registry.py`) with a content-addressable digest.

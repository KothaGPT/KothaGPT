# Base Model — Implementation Plan

Goal: build the KothaGPT base model — a Bangla-first decoder-only causal
language model in PyTorch — and a production training framework around it:
config system, checkpointing, distributed + mixed-precision training, gradient
accumulation/checkpointing, and training monitoring.

Guiding principles:

- Build on what exists: the frozen tokenizer (`ml/tokenizer/artifacts/best/`),
  the normalized/processed corpus (`data/processed/<version>/`), and the
  existing config sketch (`ml/configs/small.yaml`). No rewrite.
- Everything ships as a `torch.nn.Module` with unit tests; correctness is
  verified on toy inputs before any GPU run.
- Training runs are reproducible: every artifact is tagged with model config
  digest + data version + tokenizer digest.
- PyTorch is a new training-only dependency; it must not leak into the API
  service requirements (`services/api/requirements.txt`).

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Architecture | `ml/configs/small.yaml` declares `architecture: decoder_transformer` | no written decision, no module code |
| Model package | `ml/models/` empty (`.gitkeep`) | nothing |
| Training entrypoint | `ml/pretrain/train.py` scaffold (raises on run) | no dataset loading, model construction, loop |
| Config system | `ml/configs/small.yaml` (model/training/data) | no loader/validator/serialization |
| Tokenizer | frozen `ml/tokenizer/artifacts/best/` (BPE, GPT-2-style `▁`) | vocab size not yet wired into model config |
| Data | `data/pipeline` + `data/processed/<version>/{train,validation}` | no streaming dataset/DataLoader shim |
| PyTorch | not installed anywhere | new training-only dependency required |
| Checkpoints / distributed / mixed precision / monitoring | none | whole workstream missing |

---

## Workstreams

### WS-1 — Transformer architecture নির্বাচন (`ml/models/DECISION.md`)

Goal: fix and document the architecture before writing code.

- Formalize the decoder-only, causal-LM choice already sketched in
  `small.yaml`; evaluate and reject the alternatives in writing:
  encoder-decoder (no need for seq2seq in KothaGPT v1) and encoder-only.
- Fix the block micro-architecture:
  - pre-LayerNorm / pre-RMSNorm (stable at depth),
  - SwiGLU feed-forward (stronger than plain ReLU MLP per token budget),
  - RoPE (rotary positional embeddings) — extrapolates better than learned
    absolute positions and needs no learned position table,
  - optional ALiBi as a documented fallback if RoPE extrapolation underperforms
    on long-context evals.
- Define the target shape constants from `small.yaml` (hidden 768, layers 12,
  heads 12, ctx 4096) and set `vocab_size` from the frozen tokenizer.
- Deliverables: `ml/models/DECISION.md`, updated `ml/configs/*.yaml`.
- Metric: decision doc records rationale + rejected alternatives; config yaml
  round-trips through the loader (WS-3) with zero drift.

### WS-2 — PyTorch ভিত্তিক training framework তৈরি (`ml/trainer/`)

Goal: a real, runnable training framework replacing the `train.py` scaffold.

- Add PyTorch as a training-only dependency (`ml/requirements.txt`, torch>=2.x,
  CPU-wheels default; CUDA/ROCm builds documented in README). Keep out of
  `services/api/requirements.txt`.
- New package `ml/trainer/` with a clean split:
  - `train.py` — orchestrator CLI (subcommand `run`), mirrors tokenizer CLI
    style (`python -m ml.trainer.cli run --config ...`).
  - `dataset.py` — shard-aware streaming dataset + `DataLoader` over
    `data/processed/<version>/train`; yields token ids for causal LM.
  - `loop.py` — the training/validation loop (WS-7/8 features plug in here).
  - `scheduler.py` — cosine-with-warmup LR schedule.
  - `evaluate.py` — held-out perplexity + greedy sample generation for smoke
    checks.
- `ml/pretrain/train.py` becomes a thin wrapper calling `ml.trainer`.
- Deliverables: `ml/trainer/*`, `ml/requirements.txt`, `ml/pretrain/train.py`
  wired to it.
- Metric: `python -m ml.trainer.cli run --config ml/configs/small.yaml
  --max-steps 10` completes on CPU end-to-end (fit on a toy shard).

### WS-3 — Model configuration system তৈরি (`ml/models/config.py`)

Goal: typed, validated, serializable configuration for model + training + data.

- `ModelConfig` (dataclass + `validate`): `vocab_size, hidden_size, num_layers,
  num_heads, intermediate_size, max_position_embeddings, rms_norm_eps,
  tie_word_embeddings, rope_theta, ...`. `intermediate_size` derived (e.g.
  8/3 × hidden) or explicit.
- `TrainingConfig`: `batch_size, gradient_accumulation_steps, learning_rate,
  max_steps, warmup_steps, mixed_precision, seed, grad_clip, ...`.
- `DataConfig`: `train, validation` paths + tokenizer artifact path.
- Loader: YAML -> typed configs; `to_dict()` for JSON embedding in checkpoints.
- Cross-validation: `vocab_size` must match frozen tokenizer vocab; ctx ≤
  `max_position_embeddings`.
- Deliverables: `ml/models/config.py`, loader, unit tests.
- Metric: `small.yaml` loads and validates; a config digest (`sha256` of the
  canonical JSON) is stable and recorded in checkpoints (WS-6).

### WS-4 — Core modules (`ml/models/`)

Goal: the individual transformer building blocks, each a testable `nn.Module`.

- **Embedding layer** (`layers.py`): token embedding + (if learned positions
  are used) position embedding; weight-tied with the output head.
- **Normalization** (`layers.py`): `RMSNorm` (with `rms_norm_eps`) as the
  default; `LayerNorm` fallback selectable via config.
- **Positional encoding** (`layers.py`): RoPE (precompute cos/sin cache up to
  `max_position_embeddings`, `rope_theta` config) applied in attention.
- **Attention layer** (`attention.py`): multi-head causal self-attention with
  RoPE; efficient memory layout (head-dim-last) compatible with the standard
  `nn.functional.scaled_dot_product_attention` path (flash/mem-efficient when
  available) and a reference eager implementation behind a flag.
- **Feed-forward network** (`layers.py`): SwiGLU MLP (gate/proj) with the
  config's `intermediate_size`; optional SiLU-only variant.
- **Transformer blocks** (`blocks.py`): pre-norm residual block composing
  attention + FFN with the configured norm; optional bias-less linear layers
  (matching modern GPT configs).
- Deliverables: `ml/models/{__init__,config,layers,attention,blocks}.py`.
- Metric: unit tests per module (shapes, determinism, gradient flow,
  masked-position correctness, RMSNorm scale invariance, RoPE rotation).

### WS-5 — Output head + full model (`ml/models/model.py`)

Goal: assemble the causal-LM model.

- `KothaGPT` module: `embedding → N × TransformerBlock → final norm → output
  head (lm_head)`.
- Forward returns logits (optionally only the last position for efficiency);
  `loss` computed as cross-entropy over the sequence when labels given.
- Weight tying between token embedding and `lm_head` (config-controlled).
- `generate()` greedy loop for smoke/eval sampling; decode via the frozen
  tokenizer.
- Deliverables: `ml/models/model.py`, `ml/models/__init__.py` re-exports.
- Metric: forward/backward on a 2-block toy model; loss decreases on a
  synthetic repeating sequence; logits shape `[B, T, vocab_size]`.

### WS-6 — Model checkpoint system (`ml/trainer/checkpoint.py`)

Goal: resumable, atomic, versioned training state.

- Contents: model weights + `ModelConfig` JSON + tokenizer digest + data digest
  + optimizer state + LR scheduler state + step/epoch/global-token counters.
- Atomic writes: serialize to temp file, rename into place; keep last-N
  checkpoints plus a periodic `best` (lowest validation perplexity).
- `resume_from` restores everything above; a `--fresh` flag forces restart.
- Checkpoint layout:
  `ml/pretrain/artifacts/<run-id>/checkpoints/step-<N>.pt` (weights) +
  `config.json` + `metadata.json` (digests, counters).
- Deliverables: `checkpoint.py`, `save()`/`load()`/`resume()` + tests.
- Metric: save→fresh process→resume reproduces the exact next step (loss
  matches within fp noise); corruption-safe (partial file never loaded).

### WS-7 — Distributed training support (`ml/trainer/`)

Goal: scale from single-GPU to multi-node.

- `DistributedConfig`: `world_size, rank, master_addr/port, backend,
  use_fsdp` + CLI flags mirroring `torchrun`.
- Default: `DistributedDataParallel` for hidden 768 / 12-layer scale; FSDP
  (`shard_grad_op` → `full_shard`) behind `use_fsdp` for larger configs.
- Correctness: seed sync, deterministic DataLoader sharding, `device` from
  `local_rank`, rank-0-only logging/checkpointing, loss averaging across ranks.
- Guard with `--standalone` so CPU/dev testing never needs a launcher.
- Deliverables: DDP/FSDP wiring + multi-process tests (2×CPU workers).
- Metric: 2-process training step-loss matches single-process loss at same seed;
  only rank 0 writes checkpoints/logs.

### WS-8 — Mixed precision + gradient accumulation + gradient checkpointing

Goal: memory/speed features behind config flags.

- **Mixed precision** (`mixed_precision: bf16|fp16|none`): `torch.autocast`
  forward + `GradScaler` for fp16; bf16 default per `small.yaml`.
- **Gradient accumulation** (`gradient_accumulation_steps`): accumulate
  gradients over micro-batches, normalize loss by accumulated steps, step
  optimizer once per macro-batch; seamless with the LR scheduler.
- **Gradient checkpointing** (`gradient_checkpointing: true`): wrap blocks with
  `torch.utils.checkpoint` (recompute on backward), toggleable at runtime.
- Deliverables: loop.py flags + tests asserting exact-step parity with the
  accumulation=1 reference at fixed seed (bf16 tolerance).
- Metric: accumulated run produces bitwise-equal gradients to a single larger
  batch; checkpointing run matches eager loss within fp noise.

### WS-9 — Training monitoring (`ml/trainer/monitor.py`)

Goal: observe and steer training.

- Track per-step/per-macro-step: loss, grad norm, LR, tokens/sec, throughput,
  memory (CUDA RSS), and gradient checkpointing/skip rates.
- Backends: built-in console + JSON-lines history file (always on); optional
  WandB/`torch.utils.tensorboard` behind `--monitor-backend`.
- Periodic eval: validation perplexity every N steps; sample generation every M
  steps written to the run directory.
- Deliverables: `monitor.py`, `ml/trainer/artifacts/<run-id>/` layout
  (`history.jsonl`, `events.*`, `samples/`).
- Metric: a 100-step CPU run produces a complete `history.jsonl` with loss
  monotonically trending down on toy data and all fields present.

---

## Sequencing & dependencies

```
WS-1 architecture decision
  └─> WS-3 config system ──────────────┐
  └─> WS-4 core modules ───────────────┼─> WS-5 full model
  └─> WS-2 training framework ─────────┘         │
        │                                        │
        ├──> WS-6 checkpointing ──────── WS-7 distributed
        └──> WS-8 precision/accum/ckpt ── WS-9 monitoring
```

WS-1 → WS-3 → WS-4 → WS-5 are the critical path for a runnable model.
WS-2 and WS-6-9 layer the training framework on top; WS-7/8 build on WS-6.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Transformer architecture নির্বাচন | WS-1 |
| PyTorch ভিত্তিক training framework তৈরি | WS-2 |
| Model configuration system তৈরি | WS-3 |
| Embedding layer | WS-4 |
| Attention layer | WS-4 |
| Transformer blocks | WS-4 |
| Feed-forward network | WS-4 |
| Normalization | WS-4 |
| Positional encoding | WS-4 |
| Output head | WS-5 |
| Model checkpoint system | WS-6 |
| Distributed training support | WS-7 |
| Mixed precision training | WS-8 |
| Gradient accumulation | WS-8 |
| Gradient checkpointing | WS-8 |
| Training monitoring | WS-9 |

## Tests

```bash
python -m pytest tests/test_model_config.py   # WS-3
python -m pytest tests/test_model_modules.py  # WS-4
python -m pytest tests/test_model.py          # WS-5
python -m pytest tests/test_checkpoint.py     # WS-6
python -m pytest tests/test_distributed.py    # WS-7
python -m pytest tests/test_trainer.py        # WS-8/9 smoke
```

Never commit model weights or training artifacts; `ml/pretrain/artifacts/` is
git-ignored.
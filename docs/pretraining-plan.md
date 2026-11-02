# Pre-Training — Implementation Plan

Goal: take the frozen tokenizer + processed corpus and the `ml/trainer`
framework (built in `docs/base-model-plan.md`) and run real pre-training of the
KothaGPT base model — from a small smoke run on CPU to a large-scale,
long-context, multi-GPU training job with monitoring, resume, and evaluation.

Guiding principles:

- Build on what exists: `ml/trainer/` already implements the loop, mixed
  precision, gradient accumulation/checkpointing, DDP/FSDP, checkpoints,
  scheduler, monitoring, and eval. This plan layers *pre-training* concerns on
  top: tokenized shards, a scalable DataLoader, the GPU environment, run
  orchestration, long-context, and benchmark evaluation.
- Training runs are reproducible and resumable: every run is pinned to
  (model config digest, data version, tokenizer digest) and recorded in
  `ml/pretrain/artifacts/<run-id>/`.
- Correctness first: toy CPU smoke runs gate every GPU change; losses must
  trend down before scaling.
- PyTorch stays a training-only dependency; never import `torch` from the API
  service.

---

## Current state (map of what exists)

| Area | Exists | Gaps |
| --- | --- | --- |
| Model | `ml/models/` full KothaGPT (embedding, attention, blocks, SwiGLU, RoPE, RMSNorm, output head) | vocab_size in `small.yaml` (50000) not yet wired to frozen 16k tokenizer |
| Trainer core | `ml/trainer/` loop, checkpoint (atomic/resume), scheduler (cosine+warmup), monitor (JSONL), DDP/FSDP, mixed precision, grad accumulation/checkpointing | data loader builds all blocks eagerly in RAM; no tokens/sec throughput |
| Dataset loading | `trainer/dataset.py` `build_blocks()` + `CausalLMDataset` over raw shards | loads whole corpus into one `LongTensor`; `num_workers=0`; no memmap; not shard-scalable |
| Tokenizer | frozen 16k BPE (`ml/tokenizer/artifacts/best/`, GPT-2-style `▁`) | tokenized artifact now exists (`data/tokenized/...`) |
| Corpus | `data/processed/48245cff8e8e/` current (152M, `train/` 2 gz shards + `validation/`) | no token-id shards; no shard MANIFEST |
| Eval | `trainer/evaluate.py` (val ppl + greedy samples); `evals/run.py` + `data/benchmarks/bangla/v1` (QA/translation/summarization/generation) | no scheduled eval-on-checkpoint in the run |
| GPU env | CLI accepts `--device cuda`, bf16/fp16, DDP/FSDP wiring | no verified GPU environment, no smoke config, no torchrun orchestration |
| Long context | ctx 4096 via RoPE (`max_position_embeddings`) | no theta scaling / context extension, no long-ctx eval |
| Artifacts | — | `ml/pretrain/artifacts/` not created yet |

---

## Workstreams

### WS-1 — Dataset tokenizer pipeline তৈরি (`ml/tokenize_shards.py`) — DONE

Goal: convert the processed corpus shards into token-id shards, once, so
training never re-tokenizes.

- Add `python -m ml.tokenize_shards` CLI taking `--corpus data/processed/<version>`
  and `--tokenizer ml/tokenizer/artifacts/best/`; write a tokenized artifact
  under `data/tokenized/<version>-<tokenizer-digest>/`.
- Format: one `.npy` (uint32, `[N, block_size]`) or contiguous uint32 bin +
  index per shard, memmap-friendly; plus `MANIFEST.json` listing shards, token
  counts, block size, corpus digest, tokenizer digest.
- Keep document-boundary packing semantics from `trainer/dataset.py`; preserve
  the fixed `validation` split untouched (never token-train on it).
- Deliverables: `ml/tokenize_shards.py`, `data/tokenized/` artifact, MANIFEST.
- Metric: tokenizing the current corpus is idempotent (digest stable); token
  count matches the raw-shard token count within packing tolerance.

Status: **done** — `ml/tokenize_shards.py` + `make data-tokenize`; current
corpus tokenized to `data/tokenized/48245cff8e8e-c6900e009522/` (71.6M tokens,
17,491 blocks; train 17,157 / validation 334). Tests in
`tests/test_tokenize_shards.py` (idempotency, block equivalence with
`build_blocks`, token-count matching) wired into CI.

### WS-2 — Training shards তৈরি (`data/tokenized/` layout)

Goal: shards sized for parallel multi-GPU ingestion.

- Shard the token stream into fixed-size chunks (target ~1–2 GB each ≈
  ~250–500M tokens), numbered `shard-<N>.npy`; keep last partial shard.
- Each shard self-describing (block size + offsets in a sidecar `.json`); total
  block count recorded in MANIFEST so `max_steps` is predictable.
- Add a `make data-tokenize` target wrapping WS-1/WS-2.
- Deliverables: sharded tokenized corpus + MANIFEST; `Makefile` target.
- Metric: `make data-tokenize` reproduces byte-identical shards (idempotent);
  a validation shard holds ≥ 5k blocks.

Status: **partial** — per-input-shard `.bin` + sidecar `.json` layout and
MANIFEST are in place (`make data-tokenize`); the current corpus is small
enough to be a single training shard per split, so the fixed-size ~1–2 GB
chunking/regrouping step is deferred until the corpus grows past that.

### WS-3 — Data loader optimize করা (`ml/trainer/dataset.py`) — DONE

Goal: keep GPU fed at scale without blowing up RAM.

- Replace eager `build_blocks` materialization with a `memmap`-backed dataset
  over tokenized shards; load one shard at a time, stream blocks.
- Add `num_workers>0` + `persistent_workers` with deterministic seeding; shard
  assignment per rank so each rank reads disjoint shards (no duplicated shard
  reads across ranks).
- Measure and log tokens/sec (WS-6); keep the existing
  `(input_ids, labels)` causal shift API so the loop is unchanged.
- Deliverables: memmap dataset + DataLoader options; tokenizer still optional
  at runtime (blocks pre-tokenized by WS-1).
- Metric: a 2-worker CPU run over 1 GB of tokenized shards uses < 4 GB RSS and
  matches the eager-loader step order at the same seed.

Status: **done** — `ShardedMemmapDataset` in `ml/trainer/dataset.py` maps each
assigned `.bin` with `np.memmap` (read-only, lazy), assigns disjoint shards
per rank (round-robin) for distributed runs, and returns the same causal-shift
`(input_ids, labels)` API. `ml/trainer/cli.py` auto-selects it when
`data.train`/`data.validation` point at a tokenized corpus (MANIFEST present),
else falls back to eager `build_blocks`. Tests in `tests/test_loader_memmap.py`
(eager-order equivalence, rank disjointness/coverage, causal shift, memmap
backing, split errors) wired into CI. Real corpus verified: 17,157 train
blocks load with rank sharding `13,452 / 3,705` across 2 ranks.

### WS-4 — GPU training environment তৈরি (`infra/`, `Makefile`)

Goal: a verified, reproducible GPU path from any clean machine.

- Pin CUDA-capable torch in `ml/requirements.txt` (e.g. `--index-url` cu121
  wheel comment + README note); add `make gpu-env` to install it.
- Add `make gpu-verify`: `torch.cuda.is_available()`, device count, NCCL
  availability, and a 10-step smoke run on a tiny config.
- Add a `gpu` profile to `docker-compose.yml` (nvidia runtime, volume for
  `data/tokenized` and `ml/pretrain/artifacts`).
- Deliverables: GPU install/verify targets, compose profile, README section.
- Metric: `make gpu-verify` passes on a CUDA machine; the smoke run reports
  `device=cuda` and real throughput.

Status: **done** — `ml/gpu_verify.py` reports CUDA devices, bf16, and NCCL/Gloo support, then runs a one-step real trainer smoke test. `make gpu-env`, `make gpu-verify`, and `make gpu-smoke` are available; Compose has an opt-in NVIDIA GPU profile with tokenized-data and artifact mounts. CPU-only environments receive an actionable failure and continue to use `make train-smoke`.

### WS-5 — Small model দিয়ে test training (`ml/configs/smoke.yaml`) — DONE

Goal: prove the whole pipeline end-to-end before scaling.

- Add `ml/configs/smoke.yaml` (e.g. hidden 192, layers 2, heads 4, ctx 512,
  vocab from tokenizer, 200 steps) runnable on CPU in < 5 min on the current
  validation shard.
- Record a smoke baseline: loss must decrease monotonically-ish over the run
  and land below a documented threshold; wire it into CI as a CPU gate.
- Deliverables: `smoke.yaml`, CI job, `docs/pretraining-plan.md` baseline row
  in a run sheet.
- Metric: `make train-smoke` (CPU) completes and produces `history.jsonl`
  with a downward loss trend; exit non-zero on divergence.

Status: **done** — `ml/configs/smoke.yaml` (hidden 192, layers 2, heads 4,
ctx 512, no explicit vocab → auto-resolved to the frozen 16k tokenizer's
16,000). Tokenized shards carry their `block_size` in the artifact dir name
(`<corpus>-<tokdigest>-b<block>`), and the CLI hard-fails when a tokenized
corpus `block_size` != `model.max_position_embeddings`. `make train-smoke`
tokenizes at ctx 512 (`data-tokenize-smoke`) then trains on CPU.

Baseline (production 16k corpus, `48245cff8e8e-c6900e009522-b512`,
memmap dataset, cpu, 200 steps): start loss 9.47 → end loss 8.46,
`val_loss=8.44 / val_ppl=4639` at step 200; below `ln(16000)=9.68` (random).
Verified end-to-end: processed corpus → frozen tokenizer → tokenize-shards →
`ShardedMemmapDataset` → KothaGPT forward/backward on CPU.

### WS-6 — Loss monitoring (`ml/trainer/monitor.py`) — DONE

Goal: steer training from loss + throughput, not eyeballs.

- Extend `Monitor` records with `tokens_per_sec`, `grad_norm`, and (on CUDA)
  peak memory RSS; keep the JSONL always-on history.
- Add optional `--monitor-backend wandb|tensorboard` (base-model WS-9 scope);
  default remains console + JSONL.
- Add a loss-trend guard: warn (and optionally abort) if smoothed loss is
  flat/upward for N consecutive macro-steps.
- Deliverables: richer `history.jsonl` fields, backend flags, trend guard.
- Metric: a 100-step CPU run emits complete records (all new fields present);
  the trend guard fires on a synthetic diverging run.

Status: **done** — every macro-step record now carries `tokens_per_sec`
(tokens in the step / wall time), `grad_norm` (returned by
`clip_grad_norm_`), and `peak_mem_mb` (`torch.cuda.max_memory_allocated` on
CUDA, else process RSS via `resource`). Loss-trend guard added:
`training.trend_guard_patience` (0 = off) + `trend_guard_action`
(`warn`|`abort`); EMA-smoothed train loss that fails to improve for
`patience` consecutive macro-steps warns, or raises `TrainingDiverged` to
abort the run non-zero. `wandb|tensorboard` backends deferred to WS-9.
Smoke (cpu, 200 steps) emits all fields (~4.6k tok/s); `test_trend_guard_aborts_on_divergence`
covers the abort path (constant 999.0 loss → fires at patience).

### WS-7 — Validation monitoring (`ml/trainer/loop.py`) — DONE

Goal: track held-out quality through the run.

- Fix a stable validation window (subset of the `validation` split, config
  `eval_batches`); run it every `eval_interval`.
- Track best validation perplexity; save a `best.pt` alias alongside periodic
  checkpoints (coordinate with WS-8).
- Log `val_loss`, `val_ppl`, and sample generations to the run dir (`samples/`).
- Deliverables: best-tracker, `best.pt`, eval schedule config fields.
- Metric: two identically-seeded runs report identical val loss; `best.pt`
  exists after any run with a validation split.

Status: **done** — `training.eval_batches` (default 20) sets a stable
deterministic validation window (`shuffle=False`, capped batches). Every eval
logs `val_loss`/`val_ppl`; when val_ppl improves, `checkpoints/best.pt` is
written atomically via the refactored `save_best_checkpoint` (full state,
resumable, and untouched by `_prune` which only globs `step-*.pt`). Eval-time
greedy samples are written to `<out>/samples/step-<N>.txt` when a tokenizer is
available. Tests: `test_best_checkpoint_and_samples` (best.pt + samples
exist) and `test_identical_seeds_identical_val_loss` (two seeded runs →
identical val_ppl).

### WS-8 — Checkpointing (`ml/trainer/checkpoint.py`)

Goal: durable, restartable training state (base-model WS-6, now exercised at
pre-training scale).

- Keep atomic temp-file + rename writes; extend prune policy to always keep
  `best` in addition to `keep_last` rolling checkpoints.
- Verify checkpoint round-trip in CI: save → fresh process → resume reproduces
  the next step's loss within bf16 noise.
- Record run identity in `metadata.json`: `run_id`, data version, shard offset.
- Deliverables: `best` retention, CI round-trip test, run_id metadata.
- Metric: round-trip test passes; `metadata.json` contains run_id + digests.

Status: **done** — checkpoint sidecars and full training-state payloads now use
atomic temp-file + rename writes with cleanup on failure. Rolling `step-*.pt`
pruning excludes `best.pt`, and checkpoint tests cover best retention plus
metadata parity. Existing resume round-trip tests cover weights, optimizer,
scheduler, RNG, and next-step loss reproducibility.

### WS-9 — Resume training (`ml/trainer/cli.py`, `--resume`)

Goal: never lose a run to a node failure or a reprioritized GPU.

- Resume restores weights, optimizer, scheduler, and step counters from the
  latest (or `--checkpoint step-N`) checkpoint; `--fresh` forces restart.
- On resume, continue at the correct tokenized-shard offset so no block is
  double-consumed or skipped.
- Document the canonical resume workflow (crash → relaunch → continue) in the
  README.
- Deliverables: `--checkpoint` selector, shard-offset resume, docs.
- Metric: resume-from-step-N then stepping twice equals a from-scratch run
  stepped to N+2 at the same seed (bf16 tolerance); `--fresh` starts over.

### WS-10 — Learning-rate scheduling (`ml/trainer/scheduler.py`) — DONE

Goal: stable, well-behaved LR curves.

- Surface `warmup_steps` and `min_lr` in `TrainingConfig` and `small.yaml`
  (currently warmup is implemented but not configurable in the yaml).
- Support a linear decay option in addition to cosine; log the schedule curve
  (`lr` vs `step`) to the run dir as a small plot.
- Metric: configurable warmup/min_lr round-trip through the loader; scheduler
  produces the documented curve; LR is logged per macro-step (already true).
- Deliverables: config fields, decay variants, curve plot.

### WS-11 — Long-context training (`ml/configs/long.yaml`) — DONE

Goal: train and evaluate beyond the 4096-token default.

- Context extension: RoPE `theta` scaling (e.g. 500k–2M) and interpolated RoPE
  options in `ModelConfig`; `max_position_embeddings` already configurable.
- Add `ml/configs/long.yaml` (ctx 16k/32k, scaled theta) and a long-context
  eval set (document-level generation / needle-style probes) in `data/benchmarks`.
- Optionally support a short→long context curriculum via the trainer (warmup
  phase at ctx 4096, then extend).
- Deliverables: theta-scaling support + config, long-ctx benchmark, curriculum
  hook.
- Metric: a model trained at ctx 4096 can extend to 16k via theta scaling with
  < 5% ppl degradation on the long-ctx eval; long.yaml round-trips.

### WS-12 — Large-scale pre-training (`ml/pretrain/` orchestration)

Goal: a real multi-node run at the target scale (1–3B params).

- Add a `make train` / `make train-<run-id>` entrypoint using `torchrun` with
  rank/world-size env passthrough (the CLI already reads `RANK`/`LOCAL_RANK`).
- Enable gradient checkpointing + bf16 + accumulation for the large config;
  scale up `ml/configs/large.yaml` (hidden 2048, layers 24, heads 16).
- Pin the run to a data version + tokenizer digest; write a `run-sheet.json`
  (start/end step, loss curve, throughput, hardware) in `ml/pretrain/artifacts/<run-id>/`.
- Release checklist: final checkpoint + `config.json` + `metadata.json` +
  eval report (WS-13) + a model card.
- Deliverables: torchrun orchestration, `large.yaml`, run bookkeeping, release
  checklist.
- Metric: a 2-node × 4-GPU run trains without divergence, hitches < 5% of
  steps, and reproduces single-node loss at the same global batch size.

### WS-13 — Model evaluation (`ml/trainer/evaluate.py` + `evals/run.py`)

Goal: measure what pre-training bought.

- `evaluate.py` already reports held-out ppl + greedy samples; extend to accept
  a checkpoint path directly (`--checkpoint step-N`).
- Run the Bangla v1 benchmarks (`data/benchmarks/bangla/v1`: QA, translation,
  summarization, generation) with `evals/run.py` against the checkpoint; publish
  the report into the run dir.
- Record baseline comparisons (random-init model, tokenizer-only baseline) so
  gains are attributable to pre-training.
- Deliverables: eval-on-checkpoint command, per-run report, baseline table.
- Metric: every release checkpoint has a ppl + benchmark report; benchmark
  scores are non-worse than the random-init baseline and improve with steps.

---

## Sequencing & dependencies

```text
WS-1 tokenize pipeline ──> WS-2 shards ──> WS-3 loader optimize
                                              │
WS-4 GPU env ─────────────────────────────────┤
WS-5 small-model test (CPU gate) ──> WS-6 loss monitor
                                        ├──> WS-7 validation monitor ─┐
                                        └──> WS-8 checkpoint ──> WS-9 resume
WS-10 LR schedule (independent, parallel with WS-6..9)
WS-11 long-context (after WS-3 loader) ──> WS-12 large-scale ──> WS-13 eval
```

WS-1 → WS-2 → WS-3 are the data critical path. WS-5 (CPU smoke) gates all GPU
work. WS-12 is the summit; WS-13 measures it. WS-10 is independent and small.

## Traceability (requested items → workstreams)

| Requested item | Workstream |
| --- | --- |
| Dataset tokenizer pipeline তৈরি | WS-1 |
| Training shards তৈরি | WS-2 |
| Data loader optimize করা | WS-3 |
| GPU training environment তৈরি | WS-4 |
| Small model দিয়ে test training | WS-5 |
| Loss monitoring | WS-6 |
| Validation monitoring | WS-7 |
| Checkpointing | WS-8 |
| Resume training | WS-9 |
| Learning-rate scheduling | WS-10 |
| Long-context training | WS-11 |
| Large-scale pre-training | WS-12 |
| Model evaluation | WS-13 |

## Tests

```bash
make data-tokenize && python -m pytest tests/test_tokenize_shards.py   # WS-1/2
python -m pytest tests/test_loader_memmap.py                            # WS-3
make gpu-verify                                                        # WS-4
make train-smoke                                                       # WS-5
python -m pytest tests/test_monitor.py                                  # WS-6/7
python -m pytest tests/test_checkpoint_resume.py                        # WS-8/9
python -m pytest tests/test_scheduler.py                                # WS-10
python -m pytest tests/test_long_context.py                             # WS-11
make train && make eval-checkpoint -- checkpoint=best                    # WS-12/13
```

Never commit model weights or tokenized shards; `ml/pretrain/artifacts/` and
`data/tokenized/` are git-ignored.

# MoE Architecture Decision Record (WS-2)

Status: **prototype** — forward/backward + load-balancing loss verified; no
production-trained MoE checkpoint, no capacity enforcement, no expert-choice
routing, and no dense-vs-MoE quality/compute comparison is checked in.

## Decision

Use a **sparse top-k Mixture-of-Experts** FFN inside the KothaGPT decoder
block as the frontier-efficiency research track. Dense models scale
predictably; MoE delivers frontier quality at a fraction of the FLOPs.

## Architecture

- `ml/models/moe/router.py` — `Router`: learned softmax gate with sparse
  top-k token-to-expert assignments and a switch-style auxiliary
  load-balancing loss (gradients flow through gate probabilities).
- `ml/models/moe/layer.py` — `SparseMoELayer`: top-k routing over SwiGLU
  `ExpertFFN` experts with weighted sum aggregation.
- `ml/models/moe/block.py` — `MoEBlock`: pre-norm transformer block with
  attention and a sparse MoE FFN replacing the dense FFN.
- `ml/models/moe/train.py` — `MoETrainer`: smoke recipe with separate
  parameter groups (MoE params get a lower LR to prevent expert collapse).

## Verified

- `tests/test_moe_router.py` — router shape, top-k, auxiliary loss, capacity
  factor validation.
- `tests/test_moe_layer.py` — sparse layer forward/backward, expert
  aggregation, empty-expert skip.
- `tests/test_moe_train.py` — `make moe-smoke` runs end-to-end and produces
  non-negative task/aux/total losses; MoE params get a lower LR.

## Open

- Capacity enforcement, shared-expert fallback, expert-choice routing.
- Dense-vs-MoE quality/compute comparison at a fixed parameter budget.
- Static routing table, expert parallelism, quantization-friendly MoE
  (Phase C — `ml/inference/moe_engine.py`, `ml/inference/expert_parallel.py`).
- Frontier model config (`ml/configs/moe.yaml`) and trained checkpoint
  (Phase D).

## Metric

MoE model achieves >= 90% of dense model quality at <= 50% of FLOPs; expert
load balance > 0.8; inference latency within 1.2x dense baseline.
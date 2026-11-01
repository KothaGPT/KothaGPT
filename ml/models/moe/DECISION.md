# MoE Architecture Decision Record

## Decision

Adopt Mixture-of-Experts (MoE) architecture as the frontier efficiency path
for KothaGPT production models.

## Rationale

- Dense models scale predictably but require O(n) FLOPs for all parameters
  per token. MoE routes each token to a subset of experts (k=2), achieving
  frontier-quality models at 30-50% of the FLOPs of equivalent dense models.
- The KothaGPT decoder-only architecture (SwiGLU, RoPE, RMSNorm) is naturally
  compatible with MoE — the FFN layer can be replaced with a sparse expert
  module without changing attention, normalization, or positional encoding.
- Existing `ModelConfig` already supports `num_experts` and `active_experts`
  fields, enabling a drop-in MoE configuration.

## Architecture

- **Router**: softmax gating network producing top-k (k=2) expert assignments
  per token. Load-balanced with auxiliary loss.
- **Experts**: SwiGLU FFN modules, each with its own weights. Shared expert
  fallback for tokens that don't match any expert.
- **Capacity factor**: 1.25× average expert capacity to prevent overflow
  while allowing some routing flexibility.
- **Training**: auxiliary loss coefficient 0.01, capacity warmup for 100 steps,
  lower learning rate for expert parameters.
- **Inference**: static routing table for hot-path optimization; expert
  parallelism for batching across requests.

## Rejected Alternatives

- **Dense scaling only**: Predictable but FLOP-expensive; cannot achieve
  frontier quality at reasonable cost.
- **Pruning/distillation**: Loses model capacity; harder to train than MoE.
  MoE provides a more direct scaling path.
- **Speculative decoding**: Complements MoE but does not address the core
  parameter efficiency problem.

## Implementation

- `ml/models/moe/layer.py`: `SparseMoELayer`, `ExpertFFN`, `Router`
- `ml/models/moe/block.py`: `MoEBlock` replacing dense FFN in transformer blocks
- `ml/models/moe/router.py`: `Router` implementation with load balancing
- `ml/configs/moe.yaml`: MoE variant config (7B total, 8 experts, 2 active)
- `ml/inference/moe_engine.py`: MoE inference engine with expert parallelism

## Status

**PROPOSED** — Research phase. Implementation pending Phase 4 of
`docs/production-model-plan.md`.

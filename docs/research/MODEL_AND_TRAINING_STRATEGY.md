# Model and Training Strategy

## Current model strategy as evidenced in the repo

The repo is not yet a fully trained base-model deployment; it is a model-architecture and training-pipeline scaffold. The strongest evidence for the current strategy is:

- `configs/model.yaml` sets `architecture: decoder_transformer` and `context_length: 4096`.
- `ml/models/config.py` defines a standard decoder transformer configuration with RoPE, RMSNorm, and a `tie_word_embeddings` option.
- `ml/models/layers.py` implements a transformer stack using rotary attention, SwiGLU MLP, and causal masking.
- `ml/tokenizer/README.md` describes a custom tokenizer training approach.
- `ml/README.md` describes a planned stack covering tokenizer, pretraining, SFT, preference tuning, and inference.

## Current strategy assessment

### 1. Continue using an existing multilingual base model
Benefits:
- quick path to product quality
- lower initial engineering cost
- easier to compare to public baselines

Limitations:
- less control over Bangla-specific robustness
- risk of weaker dialect and script handling

Recommendation: viable near-term option if the project needs fast iteration, but it should not replace eventual Bangla-specific tuning.

### 2. Bangla-oriented instruction fine-tuning
Benefits:
- focus on immediate user-facing task quality
- aligned with current dataset and tokenizer work

Limitations:
- depends on a solid corpus and evaluation suite
- not enough for deep multilingual or domain-general reasoning

Recommendation: the strongest near-term option for KothaGPT.

### 3. LoRA / QLoRA parameter-efficient fine-tuning
Benefits:
- lower hardware requirements
- faster experimentation
- suitable for research teams and small budgets

Limitations:
- less control than full fine-tuning
- may underperform on heavily domain-specific adaptation

Recommendation: practical and realistic for an early production stage.

### 4. Domain-specific adaptation
Benefits:
- stronger quality in Bangla conversational and technical domains
- easier to benchmark and validate

Limitations:
- needs curated data and policy review

Recommendation: do after establishing a broad instruction set and evaluation baseline.

### 5. Retrieval-augmented generation
Benefits:
- reduces hallucination risk
- useful for factual and document-grounded tasks

Limitations:
- requires a strong retrieval layer and evaluation harness

Recommendation: appropriate as a follow-on layer, not the primary first model strategy.

### 6. Custom tokenizer or vocabulary improvements
Benefits:
- better Bangla script coverage and token efficiency
- easier to optimize for Bangla and mixed text

Limitations:
- must be justified by measured improvement

Recommendation: continue, but only if measured token efficiency and quality gains are proven.

### 7. Training from scratch
Benefits:
- full control over data and architecture
- best long-term research path

Limitations:
- extremely expensive
- requires huge compute and data infrastructure

Recommendation: defer unless clear evidence supports a full-scale Bangla foundation model program.

## Recommended pipeline

```text
Dataset Sources
      |
      v
Data Ingestion
      |
      v
Validation & Normalization
      |
      v
Quality Filtering
      |
      v
Dataset Versioning
      |
      v
Train / Validation / Test Split
      |
      v
Tokenizer & Model Preparation
      |
      v
Fine-tuning / Adaptation
      |
      v
Evaluation
      |
      v
Model Selection
      |
      v
Model Registry
      |
      v
Inference & Monitoring
```

## Recommendation

For the next stage, the most realistic strategy is: use a verified multilingual or small open base model, add Bangla-first instruction tuning, keep a strong tokenizer and data quality pipeline, and adopt LoRA/QLoRA for a fast, measurable iteration path. Explicitly treat from-scratch pretraining as a later-stage research decision rather than the default path.

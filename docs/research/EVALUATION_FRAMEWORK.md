# Evaluation Framework

## Goal

Create a practical Bangla evaluation framework that separates model quality from benchmark hype and makes regressions visible before release.

## Existing evidence

The repo already includes a substantial evaluation harness:

- `evals/run.py`
- `evals/metrics.py`
- `evals/suites/*.yaml`
- `evals/results/*.json` and `.REPORT.md`
- `tests/test_bangla_evals.py`

This demonstrates that the project already recognizes the need for benchmark suites and reproducible evaluation.

## Recommended categories

| Category | Target |
| --- | --- |
| Language understanding | Bangla comprehension in everyday and formal contexts |
| Fluency | Natural Bangla generation |
| Grammar | Correctness and readability |
| Spelling | Robustness to common spelling variation |
| Translation | Bangla<->English quality |
| QA | Factual and grounded answers |
| Reasoning | Multi-step reasoning and task completion |
| Instruction following | Alignment with explicit user cues |
| Summarization | Bangla summary quality |
| Code switching | Bangla-English mixed conversations |
| Robustness | Misspelling, weak punctuation, mixed script |
| Safety | Refusal and harmful-content handling |
| Hallucination | Unsupported claims |
| Consistency | Multi-turn coherence |

## Test structure

Each eval should include:

- `record_id`
- `prompt`
- `context` (for grounded tasks)
- `reference`
- `task_family`
- `language`
- `split`
- `metadata`

## Metrics

Use a mix of metrics rather than a single claim:

- exact match
- F1
- ROUGE
- chrF
- BERTScore where suitable
- perplexity for generative baselines
- human preference scoring
- task-specific success rate

## Human evaluation

Human review should cover:

- correctness
- helpfulness
- safety and refusal quality
- fluency and naturalness
- factual grounding

## Regression testing

Any model or data change should be evaluated against a fixed baseline before release. The project should maintain a stable leaderboard with clear deltas, not just raw top-line numbers.

## Acceptance criteria

- Every benchmark is reproducible from source config and dataset manifest.
- Results report both task-level and family-level metrics.
- Known limitations are called out clearly.
- Safety and hallucination are measured separately from pure fluency.

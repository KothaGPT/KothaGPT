# Vocabulary Freeze Report

- **version**: `1.0.0+7bf1a6e740e3.5d53c485`
- **algorithm**: `bpe`
- **vocab size**: 16,000 (target 16,000)
- **corpus digest**: `7bf1a6e740e3`

## Coverage
- coverage: **100.00%**
- unk rate: **0.00%**
- tokens/char: **0.2729**

## Efficiency benchmark
- avg tokens/char: **0.3361**
- dev max unk rate: **0.00%**
- dev min decode fidelity: **100%**
- dev min compression vs char: **2.81**

## Metrics
| metric | value | target | status |
| --- | --- | --- | --- |
| avg tokens/char | 0.3361 | 2.0 | PASS |
| dev max unk rate | 0.0000 | 0.005 | PASS |
| dev min decode fidelity | 1.0000 | 1.0 | PASS |
| dev min compression vs char | 2.8065 | 1.0 | PASS |

Benchmark report: `ml/tokenizer/artifacts/benchmark.json`
# Dataset Improvement Plan

## Objective

Turn the existing dataset pipeline into a rigorous, reproducible Bangla data foundation that can support training runs and public evaluations without unreviewed privacy, licensing, or leakage risks.

## Workstreams

### 1. Provenance and source governance
- Add dataset manifest metadata: source, URL, collection date, license, checksum, and contact owner.
- Enforce an allowlist for approved licenses.
- Refuse ingestion when source metadata is incomplete.

### 2. Quality filtering and text normalization
- Keep current Unicode, punctuation, and script normalization.
- Add stronger domain-specific filters for conversational Bangla, code-switching, and noisy web text.
- Add a noisy-text classification score to support rejection or quarantine.

### 3. Privacy and safety
- Extend PII scanning to broader real-world patterns.
- Add unsafe content and toxicity tiers.
- Define retention and redaction policies for user-generated content.

### 4. Splitting and leakage detection
- Add an explicit train/validation/test split contract.
- Check for near-duplicate leakage across splits.
- Add file- or source-level leakage detection for benchmark contamination.

### 5. Versioning and reproducibility
- Publish every dataset build with a version ID, config digest, and source manifest.
- Store a report with counts, quality metrics, and deduplication statistics.

## Priority

P0: dataset provenance and license policy
P1: PII and filtering quality gates
P1: split and leakage protection
P2: domain and dialect metadata
P2: dataset cards and human review workflow

## Acceptance criteria

- Every ingested dataset has a source manifest and license check.
- Training data passes PII and quality filters.
- Validation and test splits are non-overlapping and versioned.
- All dataset reports are reproducible from committed config and the current source manifest.

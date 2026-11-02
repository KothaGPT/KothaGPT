# Dataset Audit

## Current state

The repo contains a real, functioning data pipeline and a clear intention to produce a legal, high-quality Bangla corpus, but the actual downloaded corpora are intentionally not committed. The data layer is well-structured for research-oriented dataset work and is more mature than most of the rest of the repo.

## Evidence

- `data/README.md` describes a raw → normalize → quality filter → deduplicate → split → version pipeline.
- `data/pipeline/cli.py` implements a fully usable pipeline CLI.
- `data/pipeline/quality.py` includes PII, length, spam, and toxic-content checks.
- `data/raw/.licenses.json` shows explicit license metadata for a domain pattern.
- `data/scripts/corpora.manifest.json` is a template for source tracking and corpus fetching.
- `tests/test_data_pipeline.py` validates normalization, PII redaction, deduplication, and split determinism.

## Strengths

- clear pipeline stages and commands
- automated filtering for Bangla language quality, length, and PII
- determinism in split logic and versioning
- support for exact and near deduplication
- explicit dataset version outputs under `data/processed/`

## Weaknesses

- no canonical dataset card or user-facing provenance document
- no explicit domain taxonomy or dialect coverage map
- licensing review is still manual and ad hoc
- no formal dataset quality score or human review workflow
- no public evaluation leakage policy between training and test data
- no dataset indexing or manifest enforcement for all corpora

## Recommended improvements

1. Maintain a central manifest of every corpus source, license, checksum, and collection date.
2. Enforce a minimum quality threshold for acceptance into the training set.
3. Add language, script, domain, and dialect metadata to every record.
4. Block PII and restricted-content sources before training ingestion.
5. Create a reproducible train/validation/test split policy with leakage checks.

## Conclusion

The dataset pipeline is one of KothaGPT’s strongest implemented foundations. The next step is not to reinvent it, but to formalize provenance, license tracking, and dataset metadata so it can safely scale beyond small research runs.

# Dataset Schema Proposal

## Objective

Standardize the dataset schema so records are consistent, auditable, and fit for training, evaluation, and safety review.

```json
{
  "id": "unique-example-id",
  "source": "dataset-source",
  "language": "bn",
  "script": "Bengali",
  "dialect": "standard",
  "task": "instruction_following",
  "instruction": "প্রশ্ন বা নির্দেশনা",
  "input": "",
  "output": "উত্তর",
  "metadata": {
    "license": "verified-license",
    "quality_score": null,
    "review_status": "pending"
  }
}
```

## Required fields

- `id`: stable record identifier
- `source`: original dataset or collection source
- `language`: primary language code (`bn`, `en`, `mixed`)
- `script`: script label such as `Bengali`, `Latin`, or `Mixed`
- `dialect`: `standard`, `formal`, `colloquial`, or `unknown`
- `task`: classification of the record type
- `instruction`, `input`, `output`: compatible with instruction-tuning tasks
- `metadata`: legal, quality, and provenance status

## Additional fields recommended

- `domain`
- `quality_flags`
- `pii_detected`
- `split`
- `checksum`
- `source_hash`
- `reviewed_by`
- `collection_date`

## Quality profile

Each record should support:

- script ratio
- dialect confidence
- toxicity score
- privacy risk flags
- duplication status
- license compliance check

## Implementation guidance

The project’s current `data/pipeline` design is compatible with this schema direction. The key gap is metadata discipline rather than transformation logic.

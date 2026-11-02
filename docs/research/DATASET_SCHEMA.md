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
    "review_status": "pending",
    "region": "bn-BD",
    "script_pattern": "bn_en",
    "phenomena": ["code_switching"],
    "family": "code_switched"
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
- `region`: locale or reviewed regional variety (for example `bn-BD`)
- `script_pattern`: dominant input-script pattern (for example `bn`, `bn_en`,
  `latin_bn`, or corresponding `*_noisy` patterns)
- `phenomena`: one or more robustness tags such as `code_switching`,
  `transliteration`, or `repeated_punctuation`
- `family`: benchmark grouping such as `formal`, `colloquial`,
  `code_switched`, or `transliterated`
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

The KGPT-005 Bangla robustness fixtures implement this taxonomy in
`data/benchmarks/bangla/v1/bangla_robustness.jsonl`. Their `dialect` values
distinguish standard from colloquial register; the initial fixtures use the
`bn-BD` locale and do not claim coverage of individually verified regional
dialects. Native-speaker review is required before making that claim.

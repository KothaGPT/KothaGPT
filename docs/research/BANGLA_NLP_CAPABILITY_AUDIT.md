# Bangla NLP Capability Audit

## Executive assessment

KothaGPT has a credible foundation in Bangla language preprocessing, tokenization, and evaluation. It does not yet show a complete end-to-end Bangla NLP stack for production-quality conversation, reasoning, or language understanding. The strongest implemented features are in text normalization, transliteration, script handling, and evaluation harnesses; the weakest areas are semantic understanding, NER, POS tagging, conversational memory, and domain-specific multilingual QA.

## Capability matrix

| Capability | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Standard Bangla text handling | IMPLEMENTED | `data/pipeline/normalize.py`, `tests/test_bangla_foundation.py` | Unicode normalization and punctuation cleanup are in place. |
| Informal / conversational Bangla | PARTIAL | `data/pipeline/mixed.py` plus transliteration utilities | Mixed-script handling exists, but there is no explicit conversational dialect model or dataset standard. |
| Colloquial Bangla and dialects | PARTIAL | `data/README.md`, `docs/bangla-normalization.md` | No formal dialect taxonomy or dialect-aware benchmark is evident. |
| Bangla script handling | IMPLEMENTED | `data/pipeline/mixed.py` and tests | Script detection and spacing logic are explicit. |
| Banglish and transliteration | IMPLEMENTED | `ml/tokenizer/transliterate.py`, `tests/test_bangla_foundation.py` | Good for script conversion and mixed text boundaries. |
| English–Bangla code-switching | PARTIAL | Mixed-script detection and transliteration utilities | Handling exists at the text layer, not as a trained downstream behavior. |
| Technical terminology | EXPERIMENTAL | `ml/tokenizer` vocabulary and benchmark suites | No dedicated terminology lexicon is evident. |
| Spelling variations | PARTIAL | Normalization pipeline | Robustness exists, but not through a formal typo-aware benchmark. |
| Unicode normalization | IMPLEMENTED | `normalize.py` and tests | Includes handling of ZWJ, ligatures, and punctuation normalization. |
| Compound words / morphology | PARTIAL | Tokenizer training and corpus pipeline | Tokenizer can help, but there is no morphological analysis layer within the project. |
| Language identification | IMPLEMENTED | `data/pipeline/quality.py` | `detect_language` and `bengali_ratio` are present. |
| Tokenization | IMPLEMENTED | `ml/tokenizer` modules and README | BPE / Unigram / WordPiece are implemented from scratch. |
| Sentence segmentation | PARTIAL | Not strongly evidenced in core modules | Not a clear first-class capability from the repo. |
| Named entity recognition | MISSING | No project module found under `ml/` or `services/` | Not implemented or documented as a supported feature. |
| Part-of-speech tagging | MISSING | No dedicated NLP pipeline evidence | Not in the repo. |
| Semantic understanding | EXPERIMENTAL | `evals/metrics.py`, `evals/suites/hallucination.yaml` | Useful for measurement, not a robust semantic model. |
| Intent classification | UNVERIFIED | No dedicated module located | Not implemented as a clear capability. |
| Question answering | PARTIAL | `evals/suites/bangla.yaml` and benchmark records | Benchmark exists, but production QA model not verifiably complete. |
| Summarization | PARTIAL | Benchmark suites exist | Evaluation coverage exists, but a stable product pipeline is not evident. |
| Translation | PARTIAL | `evals/suites/bangla.yaml` | Translation evaluation is planned and partially represented in tests. |
| Text generation | PARTIAL | Model and API scaffolding exist | Model generation is present at the architecture level; full quality not proven. |
| Retrieval-augmented generation | PARTIAL | `services/rag` and tests around RAG foundations | Integration is mentioned but not fully robustly documented. |
| Conversational memory | MISSING | No concrete memory module found | Not yet a real deliverable. |
| Safety and refusal | PARTIAL | `tests/test_security_injection.py` and `evals/suites/safety.yaml` | Better than generic; still not a mature policy system. |

## Priority recommendations

1. Improve Bangla normalization and script-robustness for mixed and noisy real-world text.
2. Add a formal evaluation suite for informal Bangla, transliterated Bangla, and code-switched conversations.
3. Add a clear dialect and domain taxonomy for user data and benchmarks.
4. Build a supported RAG pipeline with language-aware retrieval and citation discipline.
5. Only invest in advanced NER/POS/semantic components after the data and evaluation pipeline stabilizes.

## Conclusion

The project has a meaningful Bangla language foundation, especially in preprocessing and tokenization. It is not yet a fully mature Bangla NLP platform. The next emphasis should be on benchmarked real-world robustness, dialect coverage, and reliable evaluation rather than broad, unproven NLP capability expansion.

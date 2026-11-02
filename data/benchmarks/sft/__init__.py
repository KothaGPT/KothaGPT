"""WS-11: SFT instruction-following benchmark datasets (per-family).

Deterministic generators for per-family SFT eval corpora under
``data/benchmarks/sft/v1``. Records follow the ``InstructionRecord`` JSONL
schema and include deterministic record IDs and dev/test split metadata.
"""

from __future__ import annotations

from .generate import generate_all, main

__all__ = ["generate_all", "main"]
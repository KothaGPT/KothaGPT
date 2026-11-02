"""Load a tiny example Hugging Face model for the ``/v1/chat`` prototype.

The heavy ``transformers`` / ``torch`` imports are deferred until the first
load so the base ``services/api/requirements.txt`` install stays fast. To
enable the ``hf`` backend:

.. code-block:: bash

    pip install -r services/api/requirements-hf.txt
    KOTHAGPT_INFERENCE_BACKEND=hf make serve-proto

Models and tokenizers are cached under the local HF cache (see ``HF_HOME`` in
``.env.example``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_EXAMPLE_MODEL = "hf-internal-testing/tiny-random-gpt2"

_DEPS_HINT = "pip install -r services/api/requirements-hf.txt"


@dataclass
class ExampleModel:
    model: Any
    tokenizer: Any
    torch: Any


def load_example_model(model_name: str = DEFAULT_EXAMPLE_MODEL) -> ExampleModel:
    """Load ``model_name`` and return it wrapped in an :class:`ExampleModel`.

    Raises :class:`ImportError` when the optional HF deps are not installed and
    propagates any model/network errors from ``transformers``.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            f"{model_name} requires transformers and torch. Install them with: {_DEPS_HINT}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    return ExampleModel(model=model, tokenizer=tokenizer, torch=torch)

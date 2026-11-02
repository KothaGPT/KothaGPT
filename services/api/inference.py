"""Chat-generation shims for the Kotha GPT API.

Two pluggable generation strategies:

- :func:`canned_generate`: a deterministic stub that returns a canned
  generation (an improved replacement for the original one-line stub).
- :class:`HFExampleInference`: loads a tiny Hugging Face model and generates
  from the prompt, degrading to :func:`canned_generate` when the optional deps
  or the model are unavailable.

The concrete backends in :mod:`services.api.core.example_backend` plug these
shims into the API's ``Backend`` factory.
"""

from __future__ import annotations

import logging
import os

from services.api.model_loader import DEFAULT_EXAMPLE_MODEL, ExampleModel, load_example_model

logger = logging.getLogger("kothagpt.api.inference")

ENV_MODEL = "KOTHAGPT_EXAMPLE_MODEL"
ENV_MAX_NEW_TOKENS = "KOTHAGPT_MAX_NEW_TOKENS"

_CANNED_TEMPLATE = (
    "প্রোটোটাইপ উত্তর: ইনফারেন্স ব্যাকএন্ড এখনো সংযুক্ত হয়নি, তাই এটি একটি "
    "ক্যানড (canned) জেনারেশন। আপনার বার্তা: «{message}»। "
    "KOTHAGPT_BACKEND=hf সেট করে একটি ক্ষুদ্র Hugging Face মডেল ব্যবহার করতে পারবেন।"
)


def canned_generate(message: str) -> str:
    """Return a deterministic canned generation for ``message``."""
    return _CANNED_TEMPLATE.format(message=(message.strip() or "(খালি)"))


class HFExampleInference:
    """Prototype generator backed by a tiny Hugging Face model.

    Loads lazily so the base API install stays lightweight. If the optional
    deps are missing or the model cannot be loaded, generation degrades to a
    canned reply instead of failing the request.
    """

    name = "hf"

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(ENV_MODEL, DEFAULT_EXAMPLE_MODEL)
        self.max_new_tokens = max_new_tokens or int(os.getenv(ENV_MAX_NEW_TOKENS, "32"))
        self._model: ExampleModel | None = None

    def _ensure_loaded(self) -> ExampleModel:
        if self._model is None:
            self._model = load_example_model(self.model_name)
        return self._model

    def generate(self, message: str) -> str:
        try:
            loader = self._ensure_loaded()
        except ImportError:
            logger.warning("hf backend requested but transformers/torch are missing")
            return canned_generate(message)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on load failures
            logger.warning("hf backend failed to load model %s: %s", self.model_name, exc)
            return canned_generate(message)

        try:
            return self._generate(loader, message)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on generation failures
            logger.warning("hf backend generation failed: %s", exc)
            return canned_generate(message)

    def _generate(self, loader: ExampleModel, message: str) -> str:
        tokenizer = loader.tokenizer
        model = loader.model
        inputs = tokenizer(message, return_tensors="pt", truncation=True)
        with loader.torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

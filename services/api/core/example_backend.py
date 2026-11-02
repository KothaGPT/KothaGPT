from __future__ import annotations

import os

from ..api.schemas import ChatCompletionRequest
from ..inference import HFExampleInference, canned_generate
from .backend import BackendCapabilities
from .mock_backend import MockBackend

ENV_MODEL = "KOTHAGPT_EXAMPLE_MODEL"
ENV_MAX_NEW_TOKENS = "KOTHAGPT_MAX_NEW_TOKENS"


class CannedBackend(MockBackend):
    """Backend whose chat replies come from the improved canned shim.

    All non-chat surfaces (embedding, rerank, tools, agents) keep the mock
    behaviour so the whole API works with zero model weights.
    """

    capabilities = BackendCapabilities(
        chat=True,
        streaming=True,
        embeddings=True,
        rerank=True,
        tools=True,
        agents=True,
        websocket=True,
    )

    def _build_response(self, request: ChatCompletionRequest) -> str:
        user_parts = [m.content for m in request.messages if m.role == "user"]
        last_user = user_parts[-1] if user_parts else ""
        return canned_generate(last_user)


class HFExampleBackend(MockBackend):
    """Backend that routes chat generation through a tiny Hugging Face model.

    Select with ``KOTHAGPT_BACKEND=hf``. Requires the optional deps in
    ``services/api/requirements-hf.txt``; without them, or if the model fails
    to load, chat falls back to the canned shim.
    """

    capabilities = BackendCapabilities(
        chat=True,
        streaming=True,
        embeddings=True,
        rerank=True,
        tools=True,
        agents=True,
        websocket=True,
    )

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._inference = HFExampleInference(
            model_name=model_name or os.getenv(ENV_MODEL),
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else int(os.getenv(ENV_MAX_NEW_TOKENS, "32")),
        )

    def _build_response(self, request: ChatCompletionRequest) -> str:
        user_parts = [m.content for m in request.messages if m.role == "user"]
        last_user = user_parts[-1] if user_parts else ""
        return self._inference.generate(last_user)

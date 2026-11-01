from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api.schemas import ChatCompletionRequest
from .backend import BackendCapabilities
from .config import RuntimeConfig
from .mock_backend import MockBackend


class EngineBackend(MockBackend):
    """Runtime adapter for the repo's real inference engine.

    This is the first explicit bridge between the API surface and the
    torch-backed engine under ``ml/inference``. The backend is intentionally
    resilient: if the runtime is unavailable or a checkpoint is missing, it falls
    back to the deterministic mock output rather than breaking the API.
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

    def __init__(self, checkpoint: str | Path | None = None, tokenizer_path: str | Path | None = None, device: str | None = None) -> None:
        super().__init__()
        self._engine = self._load_engine(checkpoint=checkpoint, tokenizer_path=tokenizer_path, device=device)

    def _load_engine(
        self,
        checkpoint: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
    ) -> Any | None:
        try:
            from ml.inference.engine import KothaGPTEngine
        except Exception:
            return None

        cfg = RuntimeConfig.from_env()
        model_path = checkpoint or cfg.checkpoint_path
        tokenizer_cfg = tokenizer_path or cfg.tokenizer_path
        chosen_device = device or cfg.device
        model_id = cfg.model_id
        quant = cfg.quant

        try:
            return KothaGPTEngine(
                checkpoint=model_path,
                tokenizer_path=tokenizer_cfg,
                device=chosen_device,
                model_id=model_id,
                quant=quant,
            )
        except Exception:
            return None

    def _build_response(self, request: ChatCompletionRequest) -> str:
        user_parts = [m.content for m in request.messages if m.role == "user"]
        last_user = user_parts[-1] if user_parts else ""
        if self._engine is None:
            return super()._build_response(request)
        try:
            max_new_tokens = RuntimeConfig.from_env().max_new_tokens
            generated = "".join(self._engine.generate(last_user, max_new_tokens=max_new_tokens))
            if generated.strip():
                return generated.strip()
        except Exception:
            pass
        return super()._build_response(request)


__all__ = ["EngineBackend"]

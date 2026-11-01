from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..api.schemas import ChatCompletionRequest
from .backend import BackendCapabilities, BackendUnavailableError
from .config import RuntimeConfig
from .mock_backend import MockBackend
from .runtime_contract import RuntimeStatus

logger = logging.getLogger(__name__)


class EngineBackend(MockBackend):
    """Runtime adapter for the repo's real inference engine.

    Inference is unavailable until configured model weights and a tokenizer
    are loaded; failures never produce mock output.
    """

    capabilities = BackendCapabilities(
        chat=True,
        streaming=True,
        embeddings=False,
        rerank=False,
        tools=False,
        agents=False,
        websocket=True,
    )

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._engine: Any | None = None
        self._runtime_error: Exception | None = None
        self._runtime_status = RuntimeStatus.NOT_CONFIGURED
        self._runtime_notes = (
            "Production inference requires an existing checkpoint and tokenizer; no model was loaded."
        )

        config = RuntimeConfig.from_env()
        model_value = checkpoint or config.checkpoint_path
        tokenizer_value = tokenizer_path or config.tokenizer_path
        model_path = Path(model_value) if model_value else None
        tokenizer_cfg = Path(tokenizer_value) if tokenizer_value else None
        if model_path is None or tokenizer_cfg is None:
            return
        if not self._is_checkpoint(model_path) or not self._is_tokenizer(tokenizer_cfg):
            self._runtime_status = RuntimeStatus.UNAVAILABLE
            self._runtime_notes = "Configured checkpoint or tokenizer artifact is missing or invalid."
            return

        try:
            self._engine = self._load_engine(
                checkpoint=model_path,
                tokenizer_path=tokenizer_cfg,
                device=device or config.device,
            )
        except Exception as exc:
            self._runtime_error = exc
            self._runtime_status = RuntimeStatus.UNAVAILABLE
            self._runtime_notes = "Configured model artifacts could not be loaded; inference is unavailable."
            logger.exception("Could not load the configured production model runtime")
        else:
            self._runtime_status = RuntimeStatus.READY
            self._runtime_notes = "Production inference engine loaded from configured model artifacts."

    def _load_engine(
        self,
        checkpoint: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
    ) -> Any | None:
        from ml.inference.engine import KothaGPTEngine

        cfg = RuntimeConfig.from_env()
        return KothaGPTEngine(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            device=device or cfg.device,
            model_id=cfg.model_id,
            quant=cfg.quant,
        )

    @staticmethod
    def _is_checkpoint(path: Path) -> bool:
        if path.is_dir():
            return any(
                (path / "checkpoints" / filename).is_file()
                for filename in ("best.pt", "step-0000000.pt")
            )
        return path.is_file() and path.suffix.lower() not in {".yaml", ".yml"}

    @staticmethod
    def _is_tokenizer(path: Path) -> bool:
        if path.is_dir():
            path = path / "tokenizer.json"
        return path.is_file()

    @property
    def runtime_status(self) -> RuntimeStatus:
        return self._runtime_status

    @property
    def runtime_notes(self) -> str:
        return self._runtime_notes

    def _build_response(self, request: ChatCompletionRequest) -> str:
        if self._engine is None:
            raise BackendUnavailableError(self._runtime_notes) from self._runtime_error
        user_parts = [m.content for m in request.messages if m.role == "user"]
        last_user = user_parts[-1] if user_parts else ""
        try:
            max_new_tokens = RuntimeConfig.from_env().max_new_tokens
            generated = "".join(self._engine.generate(last_user, max_new_tokens=max_new_tokens))
            if generated.strip():
                return generated.strip()
            raise BackendUnavailableError("The production model returned an empty response.")
        except BackendUnavailableError:
            self._runtime_status = RuntimeStatus.DEGRADED
            self._runtime_notes = "Production inference failed; mock output is not used as a fallback."
            raise
        except Exception as exc:
            self._runtime_status = RuntimeStatus.DEGRADED
            self._runtime_notes = "Production inference failed; mock output is not used as a fallback."
            logger.exception("Production model generation failed")
            raise BackendUnavailableError(self._runtime_notes) from exc


__all__ = ["EngineBackend"]

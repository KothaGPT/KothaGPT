from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_DEFAULT_BACKEND = "mock"
_DEFAULT_EXAMPLE_MODEL = "hf-internal-testing/tiny-random-gpt2"
_DEFAULT_MODEL_ID = "kothagpt"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_QUANT = "none"
_BACKEND_RE = re.compile(r"^[a-z0-9_-]+$")
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_DEVICE_RE = re.compile(r"^[a-z0-9_:-]+$")
_ALLOWED_QUANT_LEVELS = {"none", "int8", "int4", "bf16"}
_BACKEND_ALIASES = {
    "real": "engine",
    "engine": "engine",
    "mock": "mock",
    "canned": "canned",
    "hf": "hf",
}
_SUPPORTED_BACKENDS = frozenset(_BACKEND_ALIASES.values())


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated runtime settings for the API layer.

    The config intentionally remains explicit about the backend and the inference
    runtime knobs that can be configured from the environment. These settings are
    used by the mock/canned example backends and the real engine adapter.
    """

    backend: str = _DEFAULT_BACKEND
    api_token: str | None = None
    local_mode: bool = False
    example_model: str = _DEFAULT_EXAMPLE_MODEL
    max_new_tokens: int = 32
    model_id: str = _DEFAULT_MODEL_ID
    device: str = _DEFAULT_DEVICE
    quant: str = _DEFAULT_QUANT
    checkpoint_path: str | None = None
    tokenizer_path: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        mapping = os.environ if env is None else dict(env)
        backend = (mapping.get("KOTHAGPT_BACKEND") or _DEFAULT_BACKEND).strip().lower()
        if not backend:
            raise ValueError("KOTHAGPT_BACKEND cannot be empty")
        canonical = _BACKEND_ALIASES.get(backend, backend)
        if not _BACKEND_RE.fullmatch(canonical):
            raise ValueError(
                "KOTHAGPT_BACKEND must contain only lowercase letters, digits, '_' or '-'"
            )
        if canonical not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported KOTHAGPT_BACKEND {backend!r}; expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        backend = canonical

        raw_api_token = mapping.get("KOTHAGPT_API_TOKEN")
        api_token = raw_api_token.strip() if isinstance(raw_api_token, str) and raw_api_token.strip() else None
        local_mode = (mapping.get("KOTHAGPT_LOCAL_MODE", "false").strip().lower() == "true")

        example_model = (mapping.get("KOTHAGPT_EXAMPLE_MODEL") or _DEFAULT_EXAMPLE_MODEL).strip()
        if not example_model:
            raise ValueError("KOTHAGPT_EXAMPLE_MODEL cannot be empty")

        max_new_tokens_raw = mapping.get("KOTHAGPT_MAX_NEW_TOKENS", "32")
        try:
            max_new_tokens = int(max_new_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("KOTHAGPT_MAX_NEW_TOKENS must be an integer") from exc
        if max_new_tokens <= 0:
            raise ValueError("KOTHAGPT_MAX_NEW_TOKENS must be positive")

        model_id = (mapping.get("KOTHAGPT_MODEL_ID") or _DEFAULT_MODEL_ID).strip()
        if not model_id or not _MODEL_ID_RE.fullmatch(model_id):
            raise ValueError("KOTHAGPT_MODEL_ID must be a non-empty lowercase identifier")

        device = (mapping.get("KOTHAGPT_DEVICE") or _DEFAULT_DEVICE).strip().lower()
        if not device or not _DEVICE_RE.fullmatch(device):
            raise ValueError("KOTHAGPT_DEVICE must be a non-empty lowercase device name")

        quant = (mapping.get("KOTHAGPT_QUANT") or _DEFAULT_QUANT).strip().lower()
        if quant not in _ALLOWED_QUANT_LEVELS:
            raise ValueError(
                f"KOTHAGPT_QUANT must be one of {sorted(_ALLOWED_QUANT_LEVELS)}"
            )

        checkpoint_path = mapping.get("KOTHAGPT_MODEL_CHECKPOINT")
        checkpoint_path = checkpoint_path.strip() if isinstance(checkpoint_path, str) else None
        if checkpoint_path == "":
            checkpoint_path = None

        tokenizer_path = mapping.get("KOTHAGPT_TOKENIZER_PATH")
        tokenizer_path = tokenizer_path.strip() if isinstance(tokenizer_path, str) else None
        if tokenizer_path == "":
            tokenizer_path = None

        return cls(
            backend=backend,
            api_token=api_token,
            local_mode=local_mode,
            example_model=example_model,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
            device=device,
            quant=quant,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )


def normalize_backend_name(name: str | None) -> str:
    """Normalize a backend name and reject unsupported values early."""
    if name is None:
        return RuntimeConfig.from_env().backend
    backend = str(name).strip().lower()
    if not backend:
        raise ValueError("backend name cannot be empty")
    canonical = _BACKEND_ALIASES.get(backend, backend)
    if not _BACKEND_RE.fullmatch(canonical):
        raise ValueError(
            "backend name must contain only lowercase letters, digits, '_' or '-'"
        )
    return canonical


__all__ = ["RuntimeConfig", "normalize_backend_name"]

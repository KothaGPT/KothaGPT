"""KGPT-002 — Runtime contract: explicit backend status and capability checks.

The runtime contract makes the production status of every backend explicit so
that no mock/example backend can be mistaken for a production inference
runtime. Every backend declares a ``RuntimeKind`` and a human-readable
``RuntimeStatus``; the health endpoint and ``/v1/models/backends`` surface
these facts instead of hiding them behind a generic "ok".

Capability checks are enforced at the API boundary: a request that asks for a
feature the active backend does not support is rejected with a 4xx response
rather than silently degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .config import RuntimeConfig, normalize_backend_name

if TYPE_CHECKING:
    from .backend import Backend


class RuntimeKind(str, Enum):
    """Classification of a backend's intended deployment status."""

    MOCK = "mock"
    EXAMPLE = "example"
    PRODUCTION = "production"


class RuntimeStatus(str, Enum):
    """Operational status surfaced in health and model listings."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


# Canonical mapping: backend name -> (kind, status, notes).
# ``status`` is the *default*; a backend may report ``DEGRADED`` at runtime
# (e.g. when the engine adapter cannot load a checkpoint).
BACKEND_METADATA: dict[str, tuple[RuntimeKind, RuntimeStatus, str]] = {
    "mock": (
        RuntimeKind.MOCK,
        RuntimeStatus.READY,
        "Deterministic stub. No model weights. Not suitable for production traffic.",
    ),
    "canned": (
        RuntimeKind.EXAMPLE,
        RuntimeStatus.READY,
        "Improved canned stub. No model weights. Example/demo only.",
    ),
    "hf": (
        RuntimeKind.EXAMPLE,
        RuntimeStatus.READY,
        "Tiny Hugging Face example model. Degrades to canned when deps missing.",
    ),
    "engine": (
        RuntimeKind.PRODUCTION,
        RuntimeStatus.NOT_CONFIGURED,
        "Torch-backed production engine adapter. Requires a checkpoint + tokenizer.",
    ),
}


def backend_metadata(name: str | None = None) -> tuple[RuntimeKind, RuntimeStatus, str]:
    """Return the canonical (kind, status, notes) for a backend name.

    Unknown backends default to ``RuntimeKind.EXAMPLE`` /
    ``RuntimeStatus.NOT_CONFIGURED`` so that custom registrations are visible
    rather than silently treated as production.
    """
    canonical = normalize_backend_name(name)
    return BACKEND_METADATA.get(
        canonical,
        (RuntimeKind.EXAMPLE, RuntimeStatus.NOT_CONFIGURED, "Custom backend; status unknown."),
    )


def is_production_backend(name: str | None = None) -> bool:
    """True only when the backend is explicitly classified as production."""
    kind, _, _ = backend_metadata(name)
    return kind == RuntimeKind.PRODUCTION


def require_production_backend(name: str | None = None) -> None:
    """Raise ``RuntimeError`` unless the backend is production-classified.

    Use this guard at the boundary between API handlers and any code path that
    must not run on a mock/example backend (e.g. serving real user traffic).
    """
    if not is_production_backend(name):
        kind, status, notes = backend_metadata(name)
        raise RuntimeError(
            f"Backend {normalize_backend_name(name)!r} is {kind.value} "
            f"(status={status.value}): {notes}"
        )


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    """Snapshot of the active runtime's contract, surfaced via /health."""

    backend: str
    kind: RuntimeKind
    status: RuntimeStatus
    notes: str
    model_id: str
    device: str
    quant: str
    checkpoint_path: str | None
    tokenizer_path: str | None
    max_new_tokens: int

    @classmethod
    def from_config(
        cls,
        config: RuntimeConfig | None = None,
        backend: object | None = None,
    ) -> RuntimeContract:
        config = config or RuntimeConfig.from_env()
        kind, status, notes = backend_metadata(config.backend)
        backend_status = getattr(backend, "runtime_status", None)
        backend_notes = getattr(backend, "runtime_notes", None)
        if isinstance(backend_status, RuntimeStatus):
            status = backend_status
        if isinstance(backend_notes, str):
            notes = backend_notes
        return cls(
            backend=config.backend,
            kind=kind,
            status=status,
            notes=notes,
            model_id=config.model_id,
            device=config.device,
            quant=config.quant,
            checkpoint_path=config.checkpoint_path,
            tokenizer_path=config.tokenizer_path,
            max_new_tokens=config.max_new_tokens,
        )

    def is_production(self) -> bool:
        return self.kind == RuntimeKind.PRODUCTION

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "kind": self.kind.value,
            "status": self.status.value,
            "notes": self.notes,
            "model_id": self.model_id,
            "device": self.device,
            "quant": self.quant,
            "checkpoint_path": self.checkpoint_path,
            "tokenizer_path": self.tokenizer_path,
            "max_new_tokens": self.max_new_tokens,
            "production": self.is_production(),
        }


def capability_check(backend: Backend, capability: str) -> bool:
    """Return whether ``backend`` exposes ``capability``.

    ``capability`` is one of: chat, streaming, embeddings, rerank, tools,
    agents, websocket.
    """
    caps = backend.get_capabilities()
    return bool(getattr(caps, capability, False))


def require_capability(backend: Backend, capability: str) -> None:
    """Raise ``RuntimeError`` if ``backend`` lacks ``capability``."""
    if not capability_check(backend, capability):
        raise RuntimeError(
            f"Active backend {type(backend).__name__} does not support '{capability}'"
        )


__all__ = [
    "BACKEND_METADATA",
    "RuntimeContract",
    "RuntimeKind",
    "RuntimeStatus",
    "backend_metadata",
    "capability_check",
    "is_production_backend",
    "require_capability",
    "require_production_backend",
]
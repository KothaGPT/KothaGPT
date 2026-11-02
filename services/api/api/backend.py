from fastapi import HTTPException

from ..core import backend_factory
from ..core.config import RuntimeConfig
from ..core.runtime_contract import RuntimeStatus


def get_active_backend(*capabilities: str):
    config = RuntimeConfig.from_env()
    backend = backend_factory.create(config.backend)
    backend_capabilities = backend.get_capabilities()

    for capability in capabilities:
        if not getattr(backend_capabilities, capability, False):
            raise HTTPException(
                status_code=501,
                detail=f"Backend {config.backend!r} does not support {capability!r}.",
            )

    status = getattr(backend, "runtime_status", RuntimeStatus.READY)
    if status != RuntimeStatus.READY:
        notes = getattr(backend, "runtime_notes", "Inference backend is unavailable.")
        raise HTTPException(status_code=503, detail=notes)

    return backend

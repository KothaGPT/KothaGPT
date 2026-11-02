from fastapi import APIRouter

from ...core import backend_factory
from ...core.config import RuntimeConfig
from ...core.runtime_contract import (
    RuntimeContract,
    RuntimeKind,
    backend_metadata,
)
from ..schemas import BackendInfo, BackendList, ModelList

try:
    from ml.inference.registry import get_registry
except ImportError:
    get_registry = None  # type: ignore

router = APIRouter(prefix="/v1/models", tags=["models"])


# Production-ready model families exposed via /v1/models, in canonical order.
# Tiered models (nano/small/medium/large) are registered with artifact_path and
# content-addressable digests; research (moe) models are registered but not
# surfaced here.
_AVAILABLE_FAMILIES = ["kothagpt-nano", "kothagpt-small", "kothagpt-medium",
                       "kothagpt-large", "kothagpt", "kothagpt-embed",
                       "kothagpt-rerank"]


@router.get("", response_model=ModelList)
def list_models() -> ModelList:
    from ..schemas import Model as APIModel

    # Prefer registry data; use the backend catalog only when the registry module is unavailable.
    if get_registry is not None:
        regs = {r.id: r for r in get_registry().list_models()}
        data = []
        for family in _AVAILABLE_FAMILIES:
            rec = regs.get(family)
            if rec is not None and rec.status != "research":
                data.append(
                    APIModel(
                        id=rec.id,
                        description=rec.name,
                        context_window=rec.context_window,
                        status=rec.status,
                    )
                )
        return ModelList(data=data)
    backend = backend_factory.create()
    return ModelList(data=backend.list_models())


@router.get("/backends", response_model=BackendList)
def list_backends() -> BackendList:
    """Expose every registered backend with its capabilities *and* its
    production status, so callers can distinguish a stub from a real runtime.
    """
    data: list[BackendInfo] = []
    config = RuntimeConfig.from_env()
    for name, caps in backend_factory.describe().items():
        kind, status, notes = backend_metadata(name)
        if name == config.backend:
            contract = RuntimeContract.from_config(config, backend_factory.create(name))
            status = contract.status
            notes = contract.notes
        data.append(
            {
                "name": name,
                "capabilities": caps,
                "kind": kind.value,
                "status": status.value,
                "production": kind == RuntimeKind.PRODUCTION,
                "notes": notes,
            }
        )
    return BackendList(data=data)


@router.get("/runtime")
def runtime_contract():
    """Explicit runtime contract for the active backend (KGPT-002)."""
    config = RuntimeConfig.from_env()
    contract = RuntimeContract.from_config(config, backend_factory.create(config.backend))
    return contract.as_dict()


@router.get("/capability/{capability}")
def capability_check(capability: str):
    """Return whether the active backend supports ``capability``.

    Used by clients to decide whether to call chat/embeddings/rerank/tools/
    agents/websocket endpoints against the current deployment.
    """
    from ...core.runtime_contract import capability_check as _check

    config = RuntimeConfig.from_env()
    backend = backend_factory.create(config.backend)
    supported = _check(backend, capability)
    contract = RuntimeContract.from_config(config, backend)
    return {
        "capability": capability,
        "supported": supported,
        "backend": config.backend,
        "kind": contract.kind.value,
        "status": contract.status.value,
        "notes": contract.notes,
    }

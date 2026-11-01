from fastapi import APIRouter, HTTPException

from ...core import backend_factory
from ...core.runtime_contract import (
    RuntimeKind,
    RuntimeStatus,
    backend_metadata,
)
from ..schemas import BackendInfo, BackendList, ModelList

try:
    from ml.inference.registry import get_registry
except ImportError:
    get_registry = None  # type: ignore

router = APIRouter(prefix="/v1/models", tags=["models"])


# Production-ready model families exposed via /v1/models, in canonical order.
# Tiered (nano/medium/large) and research (moe) models are registered but not
# surfaced here.
_AVAILABLE_FAMILIES = ["kothagpt", "kothagpt-small", "kothagpt-embed", "kothagpt-rerank"]


@router.get("", response_model=ModelList)
def list_models() -> ModelList:
    from ..schemas import Model as APIModel

    # Prefer registry if available (WS-9), fallback to backend mock for CI
    if get_registry is not None:
        try:
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
            if data:
                return ModelList(data=data)
        except Exception:
            pass
    backend = backend_factory.create()
    return ModelList(data=backend.list_models())


@router.get("/backends", response_model=BackendList)
def list_backends() -> BackendList:
    """Expose every registered backend with its capabilities *and* its
    production status, so callers can distinguish a stub from a real runtime.
    """
    data: list[BackendInfo] = []
    for name, caps in backend_factory.describe().items():
        kind, status, notes = backend_metadata(name)
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
    from ...core.runtime_contract import RuntimeContract

    contract = RuntimeContract.from_config()
    return contract.as_dict()


@router.get("/capability/{capability}")
def capability_check(capability: str):
    """Return whether the active backend supports ``capability``.

    Used by clients to decide whether to call chat/embeddings/rerank/tools/
    agents/websocket endpoints against the current deployment.
    """
    from ...core.runtime_contract import capability_check as _check

    backend = backend_factory.create()
    supported = _check(backend, capability)
    kind, status, notes = backend_metadata(None)
    return {
        "capability": capability,
        "supported": supported,
        "backend": backend_factory.create().__class__.__name__.lower().replace("backend", ""),
        "kind": kind.value,
        "status": status.value,
        "notes": notes,
    }

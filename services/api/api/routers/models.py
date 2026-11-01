from fastapi import APIRouter

from ...core import backend_factory
from ..schemas import BackendList, ModelList

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
    data = []
    for name, caps in backend_factory.describe().items():
        data.append({"name": name, "capabilities": caps})
    return BackendList(data=data)

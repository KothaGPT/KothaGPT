from fastapi import APIRouter

from ..backend import get_active_backend
from ..schemas import RerankRequest, RerankResponse

router = APIRouter(prefix="/v1/rerank", tags=["rerank"])


@router.post("", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    backend = get_active_backend("rerank")
    return backend.rerank(request.model, request.query, request.documents, request.top_n)

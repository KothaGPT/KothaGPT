from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from .api.routers import (
    agents_router,
    chat_router,
    embeddings_router,
    models_router,
    rerank_router,
    tools_router,
    ws_router,
)
from .core import backend_factory
from .core.config import RuntimeConfig
from .core.engine_backend import EngineBackend
from .core.example_backend import CannedBackend, HFExampleBackend
from .core.mock_backend import MockBackend
from .core.runtime_contract import RuntimeContract, RuntimeStatus
from .rate_limit import limiter

backend_factory.register("mock", MockBackend)
backend_factory.register("canned", CannedBackend)
backend_factory.register("hf", HFExampleBackend)
backend_factory.register("engine", EngineBackend)
backend_factory.register("real", EngineBackend)

app = FastAPI(
    title="Kotha GPT API",
    description="Bangla-first AI platform: chat, streaming, tools, agents, embeddings, reranking.",
    version="0.2.0",
)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(rerank_router)
app.include_router(tools_router)
app.include_router(agents_router)
app.include_router(ws_router)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per-API-key rate limiting (WS-13).

    Skipped when the API is not configured with an API token (local mode),
    mirroring ``require_api_token`` behavior. Returns 429 with ``Retry-After``
    when a key exceeds its configured window.
    """
    cfg = RuntimeConfig.from_env()
    if cfg.api_token is None and cfg.local_mode:
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    key = "global"
    if authorization.startswith("Bearer "):
        key = authorization.removeprefix("Bearer ").strip() or "global"

    allowed, retry_after = limiter.is_allowed(key)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "rate limit exceeded"},
            headers={"Retry-After": str(max(retry_after, 1))},
        )
    return await call_next(request)


@app.get("/health")
def health():
    """Surface the active runtime contract so callers can distinguish a
    mock/example backend from a production model runtime."""
    config = RuntimeConfig.from_env()
    contract = RuntimeContract.from_config(config, backend_factory.create(config.backend))
    body = {
        "status": "ok",
        "service": "kothagpt-api",
        "version": app.version,
        "runtime": contract.as_dict(),
    }
    # A degraded or unavailable backend is not a healthy service.
    if contract.status in (
        RuntimeStatus.DEGRADED,
        RuntimeStatus.UNAVAILABLE,
        RuntimeStatus.NOT_CONFIGURED,
    ):
        body["status"] = "degraded"
    if not contract.is_production():
        body["status"] = "degraded"
        body["warning"] = (
            "Active backend is not a production model runtime; "
            "set KOTHAGPT_BACKEND=engine with a checkpoint to serve real model traffic."
        )
    return body

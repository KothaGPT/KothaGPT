import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routers import (
    agents_router,
    auth_router,
    chat_router,
    conversations_router,
    datasets_router,
    embeddings_router,
    evaluations_router,
    models_router,
    rag_router,
    rerank_router,
    tools_router,
    training_router,
    usage_router,
    ws_router,
)
from .core import backend_factory
from .core.config import RuntimeConfig
from .core.engine_backend import EngineBackend
from .core.example_backend import CannedBackend, HFExampleBackend
from .core.metrics import REQUEST_COUNT, REQUEST_DURATION, metrics
from .core.mock_backend import MockBackend
from .core.runtime_contract import RuntimeContract, RuntimeStatus
from .core.tracing import trace_span
from .rate_limit import limiter

backend_factory.register("mock", MockBackend)
backend_factory.register("canned", CannedBackend)
backend_factory.register("hf", HFExampleBackend)
backend_factory.register("engine", EngineBackend)
backend_factory.register("real", EngineBackend)

app = FastAPI(
    title="Kotha GPT API",
    description="Bangla-first AI platform: chat, streaming, tools, agents, embeddings, reranking, RAG.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=RuntimeConfig.from_env().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(embeddings_router)
app.include_router(rerank_router)
app.include_router(tools_router)
app.include_router(agents_router)
app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(evaluations_router)
app.include_router(training_router)
app.include_router(usage_router)
app.include_router(rag_router)
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


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Request counting and duration tracking (WS-13 metrics)."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration = time.perf_counter() - start
        metrics.counter(REQUEST_COUNT).inc()
        metrics.histogram(REQUEST_DURATION).observe(duration)
        response.headers["X-Kotha-Request-Duration-Ms"] = str(int(duration * 1000))
    return response


@app.get("/v1/metrics")
def metrics_endpoint():
    """Expose in-process metrics snapshot."""
    return metrics.snapshot()


@app.get("/health")
def health():
    """Surface the active runtime contract so callers can distinguish a
    mock/example backend from a production model runtime."""
    with trace_span("health.check", attributes={"service": "kothagpt-api"}):
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

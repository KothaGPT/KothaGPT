from fastapi import FastAPI

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
from .core.engine_backend import EngineBackend
from .core.example_backend import CannedBackend, HFExampleBackend
from .core.mock_backend import MockBackend
from .core.runtime_contract import RuntimeContract, RuntimeKind, RuntimeStatus

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


@app.get("/health")
def health():
    """Surface the active runtime contract so callers can distinguish a
    mock/example backend from a production model runtime."""
    contract = RuntimeContract.from_config()
    body = {
        "status": "ok",
        "service": "kothagpt-api",
        "version": app.version,
        "runtime": contract.as_dict(),
    }
    # A degraded or unavailable backend is not a healthy service.
    if contract.status in (RuntimeStatus.DEGRADED, RuntimeStatus.UNAVAILABLE):
        body["status"] = "degraded"
    if not contract.is_production():
        body["status"] = "degraded"
        body["warning"] = (
            "Active backend is not a production model runtime; "
            "set KOTHAGPT_BACKEND=engine with a checkpoint to serve real model traffic."
        )
    return body

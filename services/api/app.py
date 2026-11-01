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
    return {"status": "ok", "service": "kothagpt-api", "version": app.version}

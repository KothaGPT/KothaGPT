"""packages/core — shared types and protocol objects.

Dependency-light foundation shared by the Python API and the first-class
SDKs. Import from ``kothagpt.core.types`` (or re-exported names here).
"""

from .types import (
    BackendProtocol,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Embedding,
    EmbeddingResponse,
    ModelRef,
    RerankResponse,
    RerankResult,
    RuntimeContract,
    RuntimeKind,
    RuntimeStatus,
    ToolInvokeRequest,
    ToolInvokeResponse,
    ToolSpec,
    Usage,
)

__all__ = [
    "BackendProtocol",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Embedding",
    "EmbeddingResponse",
    "ModelRef",
    "RerankResponse",
    "RerankResult",
    "RuntimeContract",
    "RuntimeKind",
    "RuntimeStatus",
    "ToolInvokeRequest",
    "ToolInvokeResponse",
    "ToolSpec",
    "Usage",
]

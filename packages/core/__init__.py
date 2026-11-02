"""packages/core — shared types and protocol objects.

Dependency-light foundation shared by the Python API and the first-class
SDKs. Import from ``kothagpt.core.types`` (or re-exported names here).
"""

from .types import (
    AgentConfig,
    APIError,
    BackendConfig,
    BackendProtocol,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DatasetRef,
    Embedding,
    EmbeddingResponse,
    EvaluationResult,
    ModelConfig,
    ModelRef,
    PaginatedResponse,
    Pagination,
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
    "APIError",
    "AgentConfig",
    "BackendConfig",
    "BackendProtocol",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "DatasetRef",
    "Embedding",
    "EmbeddingResponse",
    "EvaluationResult",
    "ModelConfig",
    "ModelRef",
    "PaginatedResponse",
    "Pagination",
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

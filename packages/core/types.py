"""packages/core — shared types and protocol objects.

Pure-stdlib data types and structural protocols that both the Python API and
the first-class SDKs (Python / TypeScript) can import without pulling in
FastAPI, pydantic, or any web framework. This keeps the surface that must stay
in sync across language boundaries small and dependency-free.

These types describe the *wire shape* of the platform (models, messages, tools,
runtime contract) and are intentionally not the same objects as the pydantic
schemas in ``services/api/api/schemas.py``; a thin mapper can convert between
them at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RuntimeKind(str, Enum):
    MOCK = "mock"
    EXAMPLE = "example"
    PRODUCTION = "production"


class RuntimeStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True)
class ModelRef:
    id: str
    name: str
    context_window: int = 8192
    status: str = "ready"
    owned_by: str = "kothagpt"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    model: str = "kothagpt"
    messages: list[ChatMessage] = field(default_factory=list)
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False


@dataclass(frozen=True)
class ChatResponse:
    id: str
    model: str
    content: str
    role: str = "assistant"
    finish_reason: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    permission: str = "read"
    enabled: bool = True


@dataclass(frozen=True)
class ToolInvokeRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolInvokeResponse:
    name: str
    result: Any


@dataclass(frozen=True)
class Embedding:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingResponse:
    model: str
    data: list[Embedding]
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class RerankResult:
    index: int
    document: str
    relevance_score: float


@dataclass(frozen=True)
class RerankResponse:
    model: str
    results: list[RerankResult]


@dataclass(frozen=True)
class RuntimeContract:
    backend: str
    kind: RuntimeKind
    status: RuntimeStatus
    notes: str
    model_id: str = "kothagpt"
    device: str = "cpu"
    quant: str = "none"
    checkpoint_path: str | None = None
    tokenizer_path: str | None = None
    max_new_tokens: int = 32

    def is_production(self) -> bool:
        return self.kind == RuntimeKind.PRODUCTION


@runtime_checkable
class BackendProtocol(Protocol):
    """Structural protocol any inference/agent backend must satisfy.

    A production deployment can supply its own backend as long as it conforms
    to this protocol; the API layer never needs to know the concrete class.
    """

    def list_models(self) -> list[ModelRef]: ...

    def chat(self, request: ChatRequest) -> ChatResponse: ...

    def embed(self, model: str, inputs: list[str]) -> EmbeddingResponse: ...

    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int | None
    ) -> RerankResponse: ...

    def list_tools(self) -> list[ToolSpec]: ...

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvokeResponse: ...


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

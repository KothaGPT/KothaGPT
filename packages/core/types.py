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


@dataclass(frozen=True)
class ModelConfig:
    """Shared model configuration used across services and SDKs."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int | None = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    quant: str = "none"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", 4 * self.hidden_size)


@dataclass(frozen=True)
class AgentConfig:
    """Shared agent configuration describing tools, memory, and limits."""

    name: str
    model: str = "kothagpt"
    instructions: str | None = None
    tools: list[str] = field(default_factory=list)
    memory_enabled: bool = True
    max_tokens_per_run: int = 4096
    max_runs_per_minute: int = 60
    temperature: float | None = 0.7


@dataclass(frozen=True)
class BackendConfig:
    """Shared backend configuration."""

    name: str
    kind: str = "example"
    status: str = "not_configured"
    production: bool = False
    capabilities: dict[str, bool] | None = None
    notes: str = ""


@dataclass(frozen=True)
class APIError:
    """Shared error shape surfaced across API and SDKs."""

    error: str
    message: str | None = None
    code: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class Pagination:
    """Shared pagination parameters."""

    page: int = 1
    page_size: int = 20


@dataclass(frozen=True)
class PaginatedResponse:
    """Shared paginated response envelope."""

    items: list[Any] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20

    @property
    def has_more(self) -> bool:
        return self.page * self.page_size < self.total


@dataclass(frozen=True)
class EvaluationResult:
    """Shared evaluation run result."""

    run_id: str
    name: str
    model: str
    benchmark: str
    status: str
    scores: dict[str, float] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class DatasetRef:
    """Shared dataset reference."""

    id: str
    name: str
    version: str = "1.0.0"
    status: str = "ready"
    records: int = 0
    languages: list[str] = field(default_factory=list)


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

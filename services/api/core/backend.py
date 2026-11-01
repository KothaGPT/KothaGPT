from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import normalize_backend_name

if TYPE_CHECKING:
    from ..api.schemas import (
        Agent,
        AgentRun,
        AgentSpec,
        ChatCompletionChunk,
        ChatCompletionRequest,
        ChatCompletionResponse,
        EmbeddingResponse,
        Model,
        RerankResponse,
        Tool,
        ToolInvokeResponse,
    )


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """The supported surface area exposed by a backend implementation."""

    chat: bool = True
    streaming: bool = True
    embeddings: bool = True
    rerank: bool = True
    tools: bool = True
    agents: bool = True
    websocket: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "chat": self.chat,
            "streaming": self.streaming,
            "embeddings": self.embeddings,
            "rerank": self.rerank,
            "tools": self.tools,
            "agents": self.agents,
            "websocket": self.websocket,
        }


class Backend(ABC):
    """Interface implemented by inference/agent backends.

    A production deployment supplies its own backend (e.g. one that talks to a
    model runtime and an agent orchestrator); the default mock backend lets the
    whole API surface be exercised without any model weights.
    """

    capabilities: BackendCapabilities = BackendCapabilities()

    def get_capabilities(self) -> BackendCapabilities:
        return self.capabilities

    @abstractmethod
    def list_models(self) -> list[Model]: ...

    @abstractmethod
    def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    @abstractmethod
    def chat_stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]: ...

    @abstractmethod
    def embed(self, model: str, inputs: list[str]) -> EmbeddingResponse: ...

    @abstractmethod
    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int | None
    ) -> RerankResponse: ...

    @abstractmethod
    def list_tools(self) -> list[Tool]: ...

    @abstractmethod
    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvokeResponse: ...

    @abstractmethod
    def create_agent(self, spec: AgentSpec) -> Agent: ...

    @abstractmethod
    def get_agent(self, agent_id: str) -> Agent: ...

    @abstractmethod
    def list_agents(self) -> list[Agent]: ...

    @abstractmethod
    def delete_agent(self, agent_id: str) -> None: ...

    @abstractmethod
    def run_agent(self, agent_id: str, message: str) -> AgentRun: ...

    @abstractmethod
    def get_run(self, run_id: str) -> AgentRun: ...

    @abstractmethod
    def run_agent_stream(self, agent_id: str, message: str) -> AsyncIterator[dict[str, Any]]: ...


class BackendUnavailableError(RuntimeError):
    """Raised when a configured backend cannot serve inference requests."""


class BackendFactory:
    """Builds the configured backend. Swap via KOTHAGPT_BACKEND env var."""

    def __init__(self) -> None:
        self._registered: dict[str, type[Backend]] = {}
        self._instances: dict[str, Backend] = {}

    def register(self, name: str, backend: type[Backend]) -> None:
        normalized = normalize_backend_name(name)
        if not isinstance(backend, type) or not issubclass(backend, Backend):
            raise TypeError(
                f"Backend registration for {normalized!r} must be a Backend subclass"
            )
        self._registered[normalized] = backend

    def create(self, name: str | None = None) -> Backend:
        chosen = normalize_backend_name(name)
        if chosen not in self._registered:
            raise ValueError(f"Unknown backend {chosen!r}; registered: {sorted(self._registered)}")
        # Backends are shared singletons per process so in-memory state
        # (e.g. agents, tool results) persists across requests.
        if chosen not in self._instances:
            self._instances[chosen] = self._registered[chosen]()
        return self._instances[chosen]

    def list(self) -> list[str]:
        return sorted(self._registered)

    def describe(self, name: str | None = None) -> dict[str, dict[str, bool]]:
        """Expose the supported feature matrix without instantiating heavy backends."""
        if name is None:
            return OrderedDict(
                (registered_name, self._registered[registered_name].capabilities.as_dict())
                for registered_name in sorted(self._registered)
            )
        normalized = normalize_backend_name(name)
        if normalized not in self._registered:
            raise ValueError(f"Unknown backend {normalized!r}; registered: {sorted(self._registered)}")
        return {normalized: self._registered[normalized].capabilities.as_dict()}

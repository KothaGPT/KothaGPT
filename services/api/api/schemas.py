from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> int:
    return int(time.time())


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=_now)
    owned_by: str = "kothagpt"
    description: str = ""
    context_window: int = 8192
    status: Literal["ready", "research", "deprecated", "offline"] | str = "ready"


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


class BackendCapability(BaseModel):
    chat: bool = True
    streaming: bool = True
    embeddings: bool = True
    rerank: bool = True
    tools: bool = True
    agents: bool = True
    websocket: bool = True


class BackendInfo(BaseModel):
    name: str
    capabilities: BackendCapability


class BackendList(BaseModel):
    object: str = "list"
    data: list[BackendInfo]


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolCallFunction(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: _id("call"))
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "kothagpt"
    messages: list[Message]
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[dict[str, Any]]
    usage: Usage | None = None


class LegacyChatRequest(BaseModel):
    message: str
    model: str = "kothagpt"


class LegacyChatResponse(BaseModel):
    model: str
    message: str
    output: str


class EmbeddingRequest(BaseModel):
    model: str = "kothagpt-embed"
    input: str | list[str]


class Embedding(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[Embedding]
    usage: Usage = Field(default_factory=Usage)


class RerankRequest(BaseModel):
    model: str = "kothagpt-rerank"
    query: str
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)


class RerankResult(BaseModel):
    index: int
    document: str
    relevance_score: float


class RerankResponse(BaseModel):
    object: str = "list"
    model: str
    results: list[RerankResult]


class ToolInvokeRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    name: str
    result: Any


class AgentSpec(BaseModel):
    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: _id("agent"))
    object: str = "agent"
    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = 0.7
    created_at: int = Field(default_factory=_now)


class AgentRunRequest(BaseModel):
    message: str


class AgentRun(BaseModel):
    id: str = Field(default_factory=lambda: _id("run"))
    object: str = "agent.run"
    agent_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    messages: list[Message] = Field(default_factory=list)
    output: str | None = None
    created_at: int = Field(default_factory=_now)
    updated_at: int = Field(default_factory=_now)


class WsEnvelope(BaseModel):
    type: Literal[
        "chat",
        "ping",
        "agents.list",
        "agents.create",
        "agents.run",
        "tools.list",
        "tools.invoke",
        "embed",
        "rerank",
        "models.list",
    ]
    id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

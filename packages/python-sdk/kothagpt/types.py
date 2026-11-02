from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


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
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatCompletionRequest(BaseModel):
    model: str = "kothagpt"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(c.message.content for c in self.choices)


class ChatChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]

    @property
    def delta(self) -> str:
        return "".join((c.get("delta") or {}).get("content", "") for c in self.choices)


class Embedding(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[Embedding]
    usage: Usage = Field(default_factory=Usage)


class RerankResult(BaseModel):
    index: int
    document: str
    relevance_score: float


class RerankResponse(BaseModel):
    object: str = "list"
    model: str
    results: list[RerankResult]


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "kothagpt"
    description: str = ""
    context_window: int


class AgentSpec(BaseModel):
    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = 0.7


class Agent(BaseModel):
    id: str
    object: str = "agent"
    name: str
    description: str | None = None
    instructions: str | None = None
    model: str = "kothagpt"
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = 0.7
    created_at: int


class AgentRun(BaseModel):
    id: str
    object: str = "agent.run"
    agent_id: str
    status: str
    messages: list[ChatMessage] = Field(default_factory=list)
    output: str | None = None
    created_at: int
    updated_at: int


def utcnow() -> int:
    return int(datetime.now(UTC).timestamp())

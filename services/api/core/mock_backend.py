from __future__ import annotations

import ast
import asyncio
import hashlib
import math
import operator
import re
from collections.abc import AsyncIterator, Iterator, MutableMapping
from datetime import UTC
from typing import Any

# Agent tool registry wired into the API tool surface. Tools registered here
# (mock built-ins + the sandboxed code execution tool) are exposed via
# ``/v1/tools/*`` and invokable through the backend contract.
from services.agents.registry import ToolRegistry, ToolSpec
from services.agents.tools import default_handler

from ..api.schemas import (
    Agent,
    AgentRun,
    AgentSpec,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Embedding,
    EmbeddingResponse,
    FunctionDefinition,
    Message,
    Model,
    RerankResponse,
    RerankResult,
    Tool,
    ToolInvokeResponse,
    Usage,
)
from .agent_store import AgentStore, StoredMapping, create_store_from_env
from .backend import Backend, BackendCapabilities

_EMBEDDING_DIM = 256

_TOKEN_RE = re.compile(r"[\w\u0980-\u09FF']+")


def _mock_tool_specs() -> list[tuple[dict[str, Any], object]]:
    """Built-in mock tools as (spec_dict, handler) pairs.

    The calculator / current_time / web_search tools preserve the historical
    mock behaviour; the code execution tool is the sandboxed implementation
    from ``services/agents/tools``.
    """
    specs: list[tuple[dict[str, Any], object]] = [
        (
            {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "e.g. (2 + 3) * 4",
                        }
                    },
                    "required": ["expression"],
                },
                "permission": "read",
                "enabled": True,
            },
            lambda expression: {
                "expression": expression,
                "value": _safe_eval(str(expression)),
            },
        ),
        (
            {
                "name": "current_time",
                "description": "Get the current UTC time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
                "permission": "read",
                "enabled": True,
            },
            lambda: {"utc": _utc_now()},
        ),
        (
            {
                "name": "web_search",
                "description": "Search the web (mock).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                "permission": "read",
                "enabled": True,
            },
            lambda query, max_results=5: {
                "query": query,
                "max_results": max_results,
                "results": [{"title": "Mock result", "snippet": "Deterministic search stub."}],
            },
        ),
    ]
    # Wire in the sandboxed code execution tool from services/agents.
    spec_dict, handler = default_handler()
    specs.append((spec_dict, handler))
    return specs


_MODELS = [
    Model(
        id="kothagpt",
        description="General-purpose Bangla chat model.",
        context_window=8192,
        status="ready",
    ),
    Model(
        id="kothagpt-small",
        description="Fast Bangla chat model for latency-sensitive apps.",
        context_window=4096,
        status="ready",
    ),
    Model(
        id="kothagpt-embed",
        description="Bangla text embedding model.",
        context_window=2048,
        status="ready",
    ),
    Model(
        id="kothagpt-rerank",
        description="Bangla query/document reranking model.",
        context_window=2048,
        status="ready",
    ),
]

_TOKEN_RE = re.compile(r"[\w\u0980-\u09FF']+")


class MockBackend(Backend):
    """Deterministic, dependency-free backend for development and tests.

    Agent and run state live in an ``AgentStore``: in memory by default, or
    Redis/PostgreSQL when ``REDIS_URL``/``DATABASE_URL`` is configured so state
    survives instance restarts on ephemeral hosts (e.g. Vercel functions).
    """

    capabilities = BackendCapabilities(
        chat=True,
        streaming=True,
        embeddings=True,
        rerank=True,
        tools=True,
        agents=True,
        websocket=True,
    )

    def __init__(self, store: AgentStore | None = None) -> None:
        store = store or create_store_from_env()
        self._agents: MutableMapping[str, Agent] = StoredMapping(store, "agents", Agent)
        self._runs: MutableMapping[str, AgentRun] = StoredMapping(store, "runs", AgentRun)
        self._registry = ToolRegistry()
        for spec_dict, handler in _mock_tool_specs():
            self._registry.register(
                ToolSpec(
                    name=spec_dict["name"],
                    description=spec_dict.get("description", ""),
                    parameters=spec_dict.get("parameters", {}),
                    permission=spec_dict.get("permission", "read"),
                    enabled=spec_dict.get("enabled", True),
                ),
                handler,
            )
        self._allowed = {spec.name for spec in self._registry.list()}

    # ---- models ---------------------------------------------------------

    def list_models(self) -> list[Model]:
        return list(_MODELS)

    # ---- chat -----------------------------------------------------------

    def _build_response(self, request: ChatCompletionRequest) -> str:
        user_parts = [m.content for m in request.messages if m.role == "user"]
        last_user = user_parts[-1] if user_parts else ""
        return _mock_reply(last_user)

    def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        content = self._build_response(request)
        prompt_tokens = _token_count(request.messages)
        completion_tokens = _token_count([Message(role="assistant", content=content)])
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=Message(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def chat_stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        content = self._build_response(request)
        # Emit one chunk per token so streaming is visibly incremental.
        tokens = _split_tokens(content)
        if not tokens:
            tokens = [content]
        for i, tok in enumerate(tokens):
            yield ChatCompletionChunk(
                model=request.model,
                choices=[{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            )
            last = i == len(tokens) - 1
            if last:
                yield ChatCompletionChunk(
                    model=request.model,
                    choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                )

    # ---- embeddings -----------------------------------------------------

    def embed(self, model: str, inputs: list[str]) -> EmbeddingResponse:
        data = [Embedding(index=i, embedding=_embed(text)) for i, text in enumerate(inputs)]
        total = sum(_token_count([Message(role="user", content=t)]) for t in inputs)
        return EmbeddingResponse(
            model=model, data=data, usage=Usage(prompt_tokens=total, total_tokens=total)
        )

    # ---- rerank ---------------------------------------------------------

    def rerank(
        self, model: str, query: str, documents: list[str], top_n: int | None
    ) -> RerankResponse:
        scored = [
            RerankResult(index=i, document=doc, relevance_score=_rerank_score(query, doc))
            for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return RerankResponse(model=model, results=scored)

    # ---- tools ----------------------------------------------------------

    def list_tools(self) -> list[Tool]:
        return [
            Tool(
                function=FunctionDefinition(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                )
            )
            for spec in self._registry.list()
        ]

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvokeResponse:
        try:
            result = self._registry.invoke(name, arguments, allowed=self._allowed)
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        return ToolInvokeResponse(name=name, result=result)

    # ---- agents ---------------------------------------------------------

    def create_agent(self, spec: AgentSpec) -> Agent:
        agent = Agent(
            name=spec.name,
            description=spec.description,
            instructions=spec.instructions,
            model=spec.model,
            tools=spec.tools,
            temperature=spec.temperature,
        )
        self._agents[agent.id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        return self._agents[agent_id]

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def delete_agent(self, agent_id: str) -> None:
        del self._agents[agent_id]

    def run_agent(self, agent_id: str, message: str) -> AgentRun:
        agent = self.get_agent(agent_id)
        system = Message(
            role="system", content=agent.instructions or "You are a helpful assistant."
        )
        user = Message(role="user", content=message)
        assistant = Message(role="assistant", content=_mock_reply(message))
        run = AgentRun(
            agent_id=agent_id,
            status="completed",
            messages=[system, user, assistant],
            output=assistant.content,
        )
        self._runs[run.id] = run
        return run

    def get_run(self, run_id: str) -> AgentRun:
        return self._runs[run_id]

    async def run_agent_stream(self, agent_id: str, message: str) -> AsyncIterator[dict[str, Any]]:
        agent = self.get_agent(agent_id)
        output = _mock_reply(message)
        run = AgentRun(
            agent_id=agent_id,
            status="running",
            messages=[
                Message(
                    role="system", content=agent.instructions or "You are a helpful assistant."
                ),
                Message(role="user", content=message),
            ],
        )
        self._runs[run.id] = run
        yield {"event": "run.created", "run": run.model_dump()}
        for tok in _split_tokens(output):
            await asyncio.sleep(0.02)
            yield {"event": "run.delta", "delta": tok}
        run.status = "completed"
        run.output = output
        run.messages.append(Message(role="assistant", content=output))
        import time as _time_mod

        run.updated_at = int(_time_mod.time())
        self._runs[run.id] = run
        yield {"event": "run.completed", "output": output}


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expression: str) -> float | str:
    if len(expression) > 200:
        return "error: expression too long"
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
        value = _eval_node(tree.body, depth=0)
        if not math.isfinite(float(value)) or abs(value) > 1_000_000_000:
            return "error: result out of bounds"
        return value
    except (ArithmeticError, SyntaxError, ValueError, TypeError) as exc:
        return f"error: {exc}"


def _eval_node(node: ast.AST, depth: int) -> int | float:
    if depth > 20:
        raise ValueError("expression too deep")
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if abs(node.value) > 1_000_000:
            raise ValueError("number out of bounds")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent out of bounds")
        return _BINARY_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def _token_count(messages: list[Message]) -> int:
    return sum(max(1, len(_split_tokens(m.content))) for m in messages)


def _split_tokens(text: str) -> list[str]:
    # Cheap whitespace/word split; not the real tokenizer, just for mock streams.
    return [t for t in re.split(r"(\s+)", text) if t]


def _mock_reply(user_message: str) -> str:
    base = "এটি একটি মক প্রতিক্রিয়া।"
    if user_message.strip():
        base += f' আপনার বার্তা: "{user_message[:80]}"'
    base += " প্রোডাকশন ব্যাকএন্ড সংযুক্ত হলে এখানে মডেল আউটপুট আসবে।"
    return base


def _embed(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    vec = []
    for i in range(dim):
        digest = hashlib.sha256(f"{i}:{text}".encode()).digest()
        val = int.from_bytes(digest[:8], "big") / (2**64) * 2 - 1
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _rerank_score(query: str, doc: str) -> float:
    q_tokens = set(_TOKEN_RE.findall(query.lower()))
    d_tokens = set(_TOKEN_RE.findall(doc.lower()))
    if not q_tokens or not d_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens) / math.sqrt(len(q_tokens) * len(d_tokens))
    ngram_bonus = 0.0
    for n in (2, 3):
        q_ngrams = {query[i : i + n] for i in range(len(query) - n + 1)}
        d_ngrams = {doc[i : i + n] for i in range(len(doc) - n + 1)}
        if q_ngrams:
            ngram_bonus += len(q_ngrams & d_ngrams) / len(q_ngrams)
    return round(min(1.0, overlap + 0.1 * ngram_bonus), 4)


__all__ = ["Backend", "MockBackend"]

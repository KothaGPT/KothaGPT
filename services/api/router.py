"""WS-5: HTTP-level model routing integration for services/api.

Exposes the ModelRouter to the API layer so callers can either let the router
pick a tier (via the ``X-Kotha-Tier: auto`` header) or pin a specific tier
(via ``X-Kotha-Tier: nano|small|medium|large``).

The router is a thin adapter over ``ml.inference.router.ModelRouter`` so the
API service never imports torch directly — routing decisions are pure
Python and work with the mock/canned/example backends as well as the real
engine backend.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ml.inference.router import (
    ComplexitySignal,
    ModelRouter,
    RouteDecision,
    TaskClass,
)
from ml.inference.router_config import load_router_config

__all__ = [
    "ComplexitySignal",
    "ModelRouter",
    "RouteDecision",
    "TaskClass",
    "classify_task",
    "get_router",
    "load_router_config",
    "resolve_tier_header",
]

# Module-level singleton so the config file is loaded once at startup.
_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Return the shared ModelRouter instance (loads router_config.yaml once)."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


# Task-class keywords used by the complexity classifier.
_TASK_HINTS: list[tuple[TaskClass, tuple[str, ...]]] = [
    (TaskClass.AUTOCOMPLETE, ("autocomplete", "complete", "finish the word")),
    (TaskClass.SHORT_CHAT, ("hi", "hello", "hey", "how are you", "whats up")),
    (TaskClass.MULTIMODAL, ("image", "photo", "picture", "video", "audio", "document", "pdf")),
    (TaskClass.FRONTIER, ("research", "frontier", "publish", "paper", "PhD", "thesis")),
    (TaskClass.REASONING, ("reason", "prove", "logic", "step by step", "why", "how", "math")),
    (TaskClass.LONG_CONTEXT, ("long", "entire", "whole document", "full text", "summary of")),
    (TaskClass.AGENT, ("agent", "tool", "function call", "execute", "run code")),
    (TaskClass.CODING_ASSIST, ("code", "python", "function", "bug", "fix", "implement", "write a")),
    (TaskClass.RAG, ("according to", "based on", "context", "document", "source")),
    (TaskClass.GENERAL_CHAT, ("chat", "talk", "tell me", "what is", "where is", "when")),
]


def classify_task(prompt: str, task_hint: str | None = None) -> TaskClass:
    """Classify a request into a TaskClass from prompt text and an optional hint.

    The hint (from the ``X-Kotha-Task`` header) takes precedence when it
    matches a known TaskClass name.
    """
    if task_hint:
        try:
            return TaskClass(task_hint.strip().lower())
        except ValueError:
            pass
    lowered = (prompt or "").strip().lower()
    for task_class, keywords in _TASK_HINTS:
        if any(kw in lowered for kw in keywords):
            return task_class
    return TaskClass.GENERAL_CHAT


def resolve_tier_header(
    x_kotha_tier: str | None,
    prompt: str,
    task_hint: str | None = None,
    expected_tokens: int = 64,
    router: ModelRouter | None = None,
) -> tuple[str, RouteDecision | None]:
    """Resolve the ``X-Kotha-Tier`` header into a tier id.

    Returns ``(tier_id, route_decision)``. When the header is ``auto`` (or
    absent), the router produces a RouteDecision and its chosen tier is
    returned. When the header names an explicit tier, that tier is returned
    verbatim and the route decision is ``None`` (caller already pinned it).
    """
    router = router or get_router()
    if x_kotha_tier:
        normalized = x_kotha_tier.strip().lower()
        if normalized == "auto":
            pass  # fall through to router
        else:
            # Explicit tier pin: validate against known tiers.
            valid = {"nano", "small", "medium", "large"}
            if normalized not in valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown tier {normalized!r}; expected one of {sorted(valid)} or 'auto'",
                )
            return normalized, None
    task_class = classify_task(prompt, task_hint)
    signal = ComplexitySignal(
        prompt_length=len(prompt or ""),
        expected_tokens=expected_tokens,
        task_class=task_class,
    )
    decision = router.route(signal)
    return decision.tier, decision


def route_request(
    prompt: str,
    *,
    x_kotha_tier: str | None = None,
    x_kotha_task: str | None = None,
    expected_tokens: int = 64,
    router: ModelRouter | None = None,
) -> dict[str, Any]:
    """Convenience helper used by tests and the chat router.

    Returns a dict with ``tier``, ``model_id``, ``confidence``,
    ``task_class``, and ``explicit`` (whether the caller pinned the tier).
    """
    tier, decision = resolve_tier_header(
        x_kotha_tier, prompt, x_kotha_task, expected_tokens, router
    )
    task_class = classify_task(prompt, x_kotha_task)
    if decision is not None:
        return {
            "tier": tier,
            "model_id": decision.model_id,
            "confidence": decision.confidence,
            "task_class": task_class.value,
            "escalation_reason": decision.escalation_reason,
            "explicit": False,
        }
    return {
        "tier": tier,
        "model_id": f"kothagpt-{tier}",
        "confidence": 1.0,
        "task_class": task_class.value,
        "escalation_reason": None,
        "explicit": True,
    }
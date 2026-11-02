"""WS-2 Evaluation surface for the web client.

In-memory evaluation catalog and static benchmark definitions so the
evaluation page works when connected to the API.

Endpoints:

- ``GET /v1/evaluations`` — list evaluation runs (paginated).
- ``GET /v1/evaluations/{id}`` — get an evaluation run.
- ``POST /v1/evaluations`` — create (queue) an evaluation run.
- ``DELETE /v1/evaluations/{id}`` — delete an evaluation run.
- ``GET /v1/benchmarks`` — list available benchmark definitions.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1", tags=["evaluations"])

_BENCHMARKS: list[dict[str, Any]] = [
    {
        "name": "bangla-qa",
        "version": "v1",
        "tasks": [
            {"id": "bangla-qa-dev", "total": 500, "dev": 500, "test": 0},
            {"id": "bangla-qa-test", "total": 500, "dev": 0, "test": 500},
        ],
    },
    {
        "name": "reasoning",
        "version": "v1",
        "tasks": [
            {"id": "reasoning-dev", "total": 200, "dev": 200, "test": 0},
            {"id": "reasoning-test", "total": 200, "dev": 0, "test": 200},
        ],
    },
    {
        "name": "coding",
        "version": "v1",
        "tasks": [
            {"id": "coding-dev", "total": 150, "dev": 150, "test": 0},
            {"id": "coding-test", "total": 150, "dev": 0, "test": 150},
        ],
    },
    {
        "name": "safety",
        "version": "v1",
        "tasks": [
            {"id": "safety-dev", "total": 120, "dev": 120, "test": 0},
        ],
    },
]

_RUNS: list[dict[str, Any]] = []


class BenchmarkDefinition(BaseModel):
    name: str
    version: str
    tasks: list[dict[str, Any]]


class EvaluationCreate(BaseModel):
    name: str
    model: str
    benchmark: str


class EvaluationRun(BaseModel):
    id: str
    name: str
    model: str
    benchmark: str
    status: str
    scores: dict[str, float] = Field(default_factory=dict)
    startedAt: str
    completedAt: str | None = None


class EvaluationList(BaseModel):
    object: str = "list"
    data: list[EvaluationRun]
    total: int


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@router.get("/benchmarks", response_model=list[BenchmarkDefinition])
def list_benchmarks() -> list[BenchmarkDefinition]:
    return [BenchmarkDefinition(**b) for b in _BENCHMARKS]


@router.get("/evaluations", response_model=EvaluationList)
def list_evaluations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> EvaluationList:
    page = _RUNS[offset : offset + limit]
    return EvaluationList(data=page, total=len(_RUNS))


@router.get("/evaluations/{eval_id}", response_model=EvaluationRun)
def get_evaluation(eval_id: str) -> EvaluationRun:
    for run in _RUNS:
        if run["id"] == eval_id:
            return run
    raise HTTPException(status_code=404, detail="Evaluation not found")


@router.post("/evaluations", response_model=EvaluationRun, status_code=201)
def create_evaluation(payload: EvaluationCreate) -> EvaluationRun:
    eval_id = f"eval_{uuid.uuid4().hex[:16]}"
    started = _now()
    run: dict[str, Any] = {
        "id": eval_id,
        "name": payload.name,
        "model": payload.model,
        "benchmark": payload.benchmark,
        "status": "queued",
        "scores": {},
        "startedAt": started,
        "completedAt": None,
    }
    _RUNS.insert(0, run)
    return run


@router.delete("/evaluations/{eval_id}", status_code=204)
def delete_evaluation(eval_id: str) -> None:
    for i, run in enumerate(_RUNS):
        if run["id"] == eval_id:
            _RUNS.pop(i)
            return
    raise HTTPException(status_code=404, detail="Evaluation not found")


__all__ = ["router"]

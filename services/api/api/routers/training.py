"""WS-2 Training job surface for the web client.

In-memory training catalog so the dashboard's training page works when
connected to the API. Real training is handled by the ``ml/`` pipeline; these
endpoints expose job lifecycle (create/start/pause/stop) and checkpoints.

Endpoints:

- ``GET /v1/training`` — list training jobs (paginated).
- ``GET /v1/training/{id}`` — get a training job.
- ``POST /v1/training`` — create a training job.
- ``PATCH /v1/training/{id}/start|pause|stop`` — lifecycle transitions.
- ``GET /v1/training/{id}/checkpoints`` — list checkpoints.
- ``GET /v1/training/{id}/logs`` — get training logs (text).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/v1/training", tags=["training"])

_JOBS: list[dict[str, Any]] = []


class TrainingJobCreate(BaseModel):
    name: str
    model: str
    dataset: str
    tokenizer: str
    config: dict[str, Any]


class TrainingJob(BaseModel):
    id: str
    name: str
    model: str
    dataset: str
    tokenizer: str
    status: str
    step: int
    totalSteps: int
    progress: float
    loss: float
    learningRate: float
    tokens: int
    gpuUtilization: float
    memoryGB: float
    precision: str
    createdAt: str
    updatedAt: str


class TrainingList(BaseModel):
    object: str = "list"
    data: list[TrainingJob]
    total: int


class Checkpoint(BaseModel):
    id: str
    step: int
    loss: float
    sizeMB: float
    createdAt: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _job_out(job: dict[str, Any]) -> dict[str, Any]:
    return job


@router.get("", response_model=TrainingList)
def list_training(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> TrainingList:
    page = _JOBS[offset : offset + limit]
    return TrainingList(data=page, total=len(_JOBS))


@router.get("/{job_id}", response_model=TrainingJob)
def get_training(job_id: str) -> TrainingJob:
    for job in _JOBS:
        if job["id"] == job_id:
            return job
    raise HTTPException(status_code=404, detail="Training job not found")


@router.post("", response_model=TrainingJob, status_code=201)
def create_training(payload: TrainingJobCreate) -> TrainingJob:
    job_id = f"train_{uuid.uuid4().hex[:16]}"
    now = _now()
    cfg = payload.config
    job: dict[str, Any] = {
        "id": job_id,
        "name": payload.name,
        "model": payload.model,
        "dataset": payload.dataset,
        "tokenizer": payload.tokenizer,
        "status": "pending",
        "step": 0,
        "totalSteps": int(cfg.get("steps", 1000)),
        "progress": 0.0,
        "loss": 0.0,
        "learningRate": float(cfg.get("learningRate", 3e-4)),
        "tokens": 0,
        "gpuUtilization": 0.0,
        "memoryGB": 0.0,
        "precision": str(cfg.get("precision", "bf16")),
        "createdAt": now,
        "updatedAt": now,
    }
    _JOBS.insert(0, job)
    return job


def _transition(job_id: str, status: str) -> TrainingJob:
    for job in _JOBS:
        if job["id"] == job_id:
            job["status"] = status
            job["updatedAt"] = _now()
            return job
    raise HTTPException(status_code=404, detail="Training job not found")


@router.patch("/{job_id}/start", response_model=TrainingJob)
def start_training(job_id: str) -> TrainingJob:
    return _transition(job_id, "running")


@router.patch("/{job_id}/pause", response_model=TrainingJob)
def pause_training(job_id: str) -> TrainingJob:
    return _transition(job_id, "paused")


@router.patch("/{job_id}/stop", response_model=TrainingJob)
def stop_training(job_id: str) -> TrainingJob:
    return _transition(job_id, "failed")


@router.get("/{job_id}/checkpoints", response_model=list[Checkpoint])
def checkpoints(job_id: str) -> list[Checkpoint]:
    if not any(j["id"] == job_id for j in _JOBS):
        raise HTTPException(status_code=404, detail="Training job not found")
    return [
        Checkpoint(
            id=f"ckpt-{job_id}-1",
            step=100,
            loss=2.45,
            sizeMB=128.0,
            createdAt=_now(),
        )
    ]


@router.get("/{job_id}/logs")
def logs(job_id: str):
    from fastapi.responses import PlainTextResponse

    if not any(j["id"] == job_id for j in _JOBS):
        raise HTTPException(status_code=404, detail="Training job not found")
    return PlainTextResponse(
        f"[train {job_id}] no logs available (training not executed through the API)\n"
    )


__all__ = ["router"]

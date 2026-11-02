"""WS-2 Dataset management for the web client.

In-memory dataset catalog (Phase 3). Real dataset ingestion is handled by the
``data/`` pipeline; this endpoint exposes a small seed catalog so the
dashboard's datasets page works when connected to the API.

Endpoints:

- ``GET /v1/datasets`` — list datasets (paginated).
- ``GET /v1/datasets/{id}`` — get a dataset.
- ``GET /v1/datasets/{id}/preview`` — preview rows.
- ``GET /v1/datasets/{id}/versions`` — version history.
- ``DELETE /v1/datasets/{id}`` — delete (seed datasets are protected).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])

_SEED: list[dict[str, Any]] = [
    {
        "id": "ds-bangla-corpus",
        "name": "Bangla General Corpus",
        "version": "1.0.0",
        "description": "Cleaned Bangla text corpus for pretraining.",
        "status": "ready",
        "languages": ["bn"],
        "records": 1_250_000,
        "tokens": 18_000_000,
        "duplicateRatio": 0.02,
        "qualityScore": 0.91,
        "piiStatus": "clean",
        "license": "CC-BY-SA-4.0",
        "trainSplit": 0.9,
        "validationSplit": 0.05,
        "testSplit": 0.05,
        "createdAt": "2026-01-10T00:00:00Z",
        "updatedAt": "2026-01-10T00:00:00Z",
    },
    {
        "id": "ds-bangla-instruction",
        "name": "Bangla Instruction Tuning Set",
        "version": "1.0.0",
        "description": "Instruction/response pairs for SFT.",
        "status": "ready",
        "languages": ["bn", "en"],
        "records": 85_000,
        "tokens": 4_200_000,
        "duplicateRatio": 0.01,
        "qualityScore": 0.88,
        "piiStatus": "clean",
        "license": "CC-BY-4.0",
        "trainSplit": 0.9,
        "validationSplit": 0.05,
        "testSplit": 0.05,
        "createdAt": "2026-02-15T00:00:00Z",
        "updatedAt": "2026-02-15T00:00:00Z",
    },
]


class DatasetList(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
    total: int


@router.get("", response_model=DatasetList)
def list_datasets(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DatasetList:
    page = _SEED[offset : offset + limit]
    return DatasetList(data=page, total=len(_SEED))


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    for ds in _SEED:
        if ds["id"] == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str) -> dict[str, Any]:
    if not any(ds["id"] == dataset_id for ds in _SEED):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "columns": ["text", "lang", "source"],
        "rows": [
            {"text": "বাংলা ভাষা সুন্দর।", "lang": "bn", "source": "seed"},
            {"text": "এটি একটি নমুনা বাক্য।", "lang": "bn", "source": "seed"},
        ],
    }


@router.get("/{dataset_id}/versions", response_model=DatasetList)
def versions_dataset(dataset_id: str) -> DatasetList:
    if not any(ds["id"] == dataset_id for ds in _SEED):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetList(data=[next(ds for ds in _SEED if ds["id"] == dataset_id)], total=1)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Seed datasets cannot be deleted through the API.",
    )


__all__ = ["router"]

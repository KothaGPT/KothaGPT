"""WS-2 Usage / billing surface for the web client.

In-memory usage rollups so the dashboard's usage cards work when connected to
the API. Real metering would be wired to request logs here.

Endpoints:

- ``GET /v1/usage/summary`` — aggregate usage over a range.
- ``GET /v1/usage/timeseries`` — per-granularity usage points.
- ``GET /v1/usage/by-model`` — usage broken down by model.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/v1/usage", tags=["usage"])


class UsageMetric(BaseModel):
    period: str
    requests: int
    inputTokens: int
    outputTokens: int
    totalTokens: int
    latencyMs: float
    errors: int


class UsageSummary(BaseModel):
    requests: int
    tokens: int
    cost: float
    errorRate: float
    breakdown: dict[str, UsageMetric]


@router.get("/summary", response_model=UsageSummary)
def summary(range_: str = Query(default="30d", alias="range")) -> UsageSummary:
    return UsageSummary(
        requests=12_340,
        tokens=4_850_000,
        cost=24.25,
        errorRate=0.001,
        breakdown={
            "kothagpt": UsageMetric(
                period=range_,
                requests=8_900,
                inputTokens=3_200_000,
                outputTokens=1_100_000,
                totalTokens=4_300_000,
                latencyMs=320.0,
                errors=5,
            ),
            "kothagpt-small": UsageMetric(
                period=range_,
                requests=3_440,
                inputTokens=400_000,
                outputTokens=150_000,
                totalTokens=550_000,
                latencyMs=110.0,
                errors=1,
            ),
        },
    )


@router.get("/timeseries", response_model=list[UsageMetric])
def timeseries(
    range_: str = Query(default="30d", alias="range"),
    granularity: str = Query(default="day"),
) -> list[UsageMetric]:
    return [
        UsageMetric(
            period=f"day-{i}",
            requests=400 + i * 10,
            inputTokens=160_000 + i * 4_000,
            outputTokens=50_000 + i * 1_000,
            totalTokens=210_000 + i * 5_000,
            latencyMs=280.0 + i * 2,
            errors=0,
        )
        for i in range(7)
    ]


@router.get("/by-model", response_model=dict[str, UsageMetric])
def by_model() -> dict[str, UsageMetric]:
    return {
        "kothagpt": UsageMetric(
            period="30d",
            requests=8_900,
            inputTokens=3_200_000,
            outputTokens=1_100_000,
            totalTokens=4_300_000,
            latencyMs=320.0,
            errors=5,
        ),
        "kothagpt-small": UsageMetric(
            period="30d",
            requests=3_440,
            inputTokens=400_000,
            outputTokens=150_000,
            totalTokens=550_000,
            latencyMs=110.0,
            errors=1,
        ),
    }


__all__ = ["router"]

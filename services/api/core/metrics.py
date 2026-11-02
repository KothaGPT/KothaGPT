"""WS-13 Request metrics — lightweight counters and timers for the API.

Dependency-free metrics collected in-memory. Suitable for local/dev and
single-instance deployments; production can swap the backend for Prometheus.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Counter:
    value: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self.value += amount


@dataclass
class Histogram:
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    )
    counts: dict[int, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def observe(self, seconds: float) -> None:
        with self._lock:
            self.sum += seconds
            self.count += 1
            for i, b in enumerate(self.buckets):
                if seconds <= b:
                    self.counts[i] = self.counts.get(i, 0) + 1
                    break
            else:
                self.counts[len(self.buckets)] = self.counts.get(len(self.buckets), 0) + 1


class MetricsRegistry:
    """In-memory metrics registry (singleton per process)."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = defaultdict(Counter)
        self._histograms: dict[str, Histogram] = defaultdict(Histogram)
        self._lock = Lock()

    def counter(self, name: str) -> Counter:
        with self._lock:
            return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        with self._lock:
            return self._histograms[name]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {k: v.value for k, v in self._counters.items()}
            histograms = {
                k: {"count": h.count, "sum": h.sum, "buckets": dict(h.counts)}
                for k, h in self._histograms.items()
            }
            return {"counters": counters, "histograms": histograms}


metrics = MetricsRegistry()

REQUEST_COUNT = "kothagpt_http_requests_total"
REQUEST_DURATION = "kothagpt_http_request_duration_seconds"


__all__ = [
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "Counter",
    "Histogram",
    "MetricsRegistry",
    "metrics",
]

"""WS-10 Abuse detection — burst + policy evasion heuristics."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class AbuseDetector:
    def __init__(self, burst_threshold: int = 20, window_sec: int = 60):
        self.burst_threshold = burst_threshold
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self.flagged: set[str] = set()

    def record(self, tenant: str, *, tokens: int = 1) -> bool:
        now = time.time()
        q = self._hits[tenant]
        q.append(now)
        # trim window
        while q and q[0] <= now - self.window:
            q.popleft()
        # burst: count
        if len(q) >= self.burst_threshold:
            self.flagged.add(tenant)
            return True
        return False

    def is_flagged(self, tenant: str) -> bool:
        return tenant in self.flagged

    def review(self, tenant: str, decision: str) -> None:
        if decision == "allow":
            self.flagged.discard(tenant)
            self._hits[tenant].clear()

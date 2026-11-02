"""WS-13 Rate limiting — token-aware, per-key, heavily simplified for CI."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        self.max_requests = max_requests
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str = "global") -> tuple[bool, int]:
        now = time.time()
        q = self._hits[key]
        # evict old
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            retry = int(q[0] + self.window - now) + 1
            return False, retry
        q.append(now)
        return True, 0

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)

limiter = RateLimiter()

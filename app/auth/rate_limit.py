"""Small in-process sliding-window rate limiter for Phase 10."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from app.auth.service import RateLimitError


class RateLimiter:
    """Protect sensitive endpoints in a single application process.

    Phase 11 can replace this store with Redis/shared infrastructure for
    multi-worker or public deployment.
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, *, limit: int, window_seconds: int) -> None:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                raise RateLimitError("too many requests; please try again later")
            events.append(now)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

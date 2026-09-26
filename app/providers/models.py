"""Models shared by the live market-data provider layer.

The models in this module are provider-neutral.  They intentionally contain
no authentication secrets or provider-specific response objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class LiveProviderHealth:
    """Safe operational state for a live market-data provider."""

    provider: str
    configured: bool
    status: str
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    latency_ms: float | None = None
    consecutive_failures: int = 0
    requests_total: int = 0
    cache_hits: int = 0
    last_error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("last_success_at", "last_failure_at"):
            value = payload[key]
            payload[key] = value.astimezone(timezone.utc).isoformat() if value else None
        return payload


class LiveProviderError(RuntimeError):
    """Controlled error raised by a live provider adapter."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int = 503,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable

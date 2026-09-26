"""Domain contracts for market data.

No provider-specific implementation belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class OHLCBar:
    """Canonical OHLC bar contract for later phases."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None


class MarketDataProvider(Protocol):
    """Common interface for historical and future live market providers."""

    def get_ohlc(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[OHLCBar]:
        """Return canonical OHLC bars for a requested range."""
        ...

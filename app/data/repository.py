
"""Clean OHLC repository interface for later backtesting and analysis."""

from __future__ import annotations

from datetime import datetime

from app.data.schema import CanonicalOHLC


class OHLCRepository:
    """In-memory repository boundary for Phase 02.

    Persistence/query optimization can evolve later without changing callers.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[CanonicalOHLC]] = {}

    def put(self, symbol: str, timeframe: str, bars: list[CanonicalOHLC]) -> None:
        self._data[(symbol.upper(), timeframe)] = list(bars)

    def get_slice(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CanonicalOHLC]:
        bars = self._data.get((symbol.upper(), timeframe), [])
        return [
            bar for bar in bars
            if (start is None or bar.timestamp >= start)
            and (end is None or bar.timestamp <= end)
        ]

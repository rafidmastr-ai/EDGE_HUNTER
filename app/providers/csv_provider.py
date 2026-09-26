"""Historical CSV provider contract.

Actual CSV discovery, parsing and validation belong to Phase 02.
"""

from __future__ import annotations

from datetime import datetime

from app.domain.market import OHLCBar


class CsvHistoricalDataProvider:
    """Placeholder provider interface for historical CSV data."""

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCBar]:
        """Raise until Phase 02 implements the data foundation."""
        raise NotImplementedError(
            "CSV OHLC loading is implemented in Phase 02 — Historical Data Foundation."
        )

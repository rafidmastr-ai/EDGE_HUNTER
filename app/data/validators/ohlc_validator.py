
"""Deterministic validation and diagnostics for canonical OHLC data."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from app.data.schema import CanonicalOHLC


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    total_rows: int = 0
    valid_rows: int = 0
    duplicate_timestamps: int = 0
    non_monotonic_rows: int = 0
    missing_rows: int = 0
    invalid_price_rows: int = 0
    invalid_ohlc_rows: int = 0
    gaps: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.invalid_price_rows == 0 and self.invalid_ohlc_rows == 0


class OHLCValidator:
    """Validate canonical OHLC records without inventing missing market data."""

    def validate(
        self,
        bars: Iterable[CanonicalOHLC],
        symbol: str,
        timeframe: str,
    ) -> DataQualityReport:
        records = list(bars)
        report = DataQualityReport(symbol=symbol, timeframe=timeframe)
        report.total_rows = len(records)

        seen: set = set()
        previous = None

        for index, bar in enumerate(records):
            if bar.timestamp in seen:
                report.duplicate_timestamps += 1
                report.errors.append(f"duplicate timestamp at row {index}: {bar.timestamp.isoformat()}")
            seen.add(bar.timestamp)

            if previous is not None and bar.timestamp <= previous:
                report.non_monotonic_rows += 1
                report.errors.append(
                    f"non-monotonic timestamp at row {index}: {bar.timestamp.isoformat()}"
                )
            previous = bar.timestamp

            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(not p.is_finite() or p <= Decimal("0") for p in prices):
                report.invalid_price_rows += 1
                report.errors.append(f"invalid price at row {index}")
                continue

            if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
                report.invalid_ohlc_rows += 1
                report.errors.append(f"impossible OHLC relationship at row {index}")
                continue

            if bar.volume is not None and (not bar.volume.is_finite() or bar.volume < 0):
                report.errors.append(f"invalid volume at row {index}")

            report.valid_rows += 1

        return report

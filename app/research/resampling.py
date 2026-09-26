"""Deterministic OHLC resampling for research timeframes."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Sequence

from app.data.schema import CanonicalOHLC


_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
}


def timeframe_seconds(timeframe: str) -> int:
    key = timeframe.upper()
    if key not in _TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported research timeframe: {timeframe!r}")
    return _TIMEFRAME_SECONDS[key]


def resample_ohlc(bars: Sequence[CanonicalOHLC], timeframe: str) -> tuple[CanonicalOHLC, ...]:
    """Aggregate canonical OHLC bars without using future observations.

    The timestamp of a resampled candle is the timestamp of its last source bar,
    i.e. the moment at which the aggregated close becomes known. This avoids
    labeling the candle with its interval start and accidentally moving a decision
    earlier than the information is actually available.
    """
    records = tuple(bars)
    seconds = timeframe_seconds(timeframe)
    if not records:
        return ()
    if seconds == 60:
        return records

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    buckets: "OrderedDict[int, list[CanonicalOHLC]]" = OrderedDict()
    for bar in records:
        timestamp = bar.timestamp
        if timestamp.tzinfo is None:
            raise ValueError("research resampling requires timezone-aware timestamps")
        utc = timestamp.astimezone(timezone.utc)
        bucket_key = int((utc - epoch).total_seconds()) // seconds
        buckets.setdefault(bucket_key, []).append(bar)

    output: list[CanonicalOHLC] = []
    for group in buckets.values():
        volumes = [bar.volume for bar in group]
        volume: Decimal | None
        if all(item is not None for item in volumes):
            volume = sum((item for item in volumes if item is not None), Decimal("0"))
        else:
            volume = None
        output.append(
            CanonicalOHLC(
                timestamp=group[-1].timestamp.astimezone(timezone.utc),
                open=group[0].open,
                high=max(bar.high for bar in group),
                low=min(bar.low for bar in group),
                close=group[-1].close,
                volume=volume,
            )
        )
    return tuple(output)


__all__ = ["resample_ohlc", "timeframe_seconds"]

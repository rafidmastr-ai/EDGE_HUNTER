"""Removal of synthetic (non-trading) filler bars from historical OHLC files.

Some historical CSV exports fill closed-market periods with flat candles whose
open = high = low = close equals the previous close (e.g. every Sunday from
00:00 UTC until the real weekly open, holidays, metals' daily break). Those rows
are not market data: indicators (ATR, EMA, RSI, ...) computed across them are
distorted after every re-open. This module drops such runs deterministically.

A bar belongs to a synthetic run when it is flat (O = H = L = C) and either
repeats the previous close or follows a gap. A run is removed only when it is
at least ``min_run_minutes`` long AND spans at least ``min_run_bars`` bars, so
isolated quiet candles of real trading are kept.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from app.data.schema import CanonicalOHLC

DEFAULT_MIN_RUN_MINUTES = 30
DEFAULT_MIN_RUN_BARS = 3


def _base_step_seconds(bars: Sequence[CanonicalOHLC]) -> float:
    deltas = [
        (bars[i].timestamp - bars[i - 1].timestamp).total_seconds()
        for i in range(1, min(len(bars), 5001))
        if bars[i].timestamp > bars[i - 1].timestamp
    ]
    return float(statistics.median(deltas)) if deltas else 60.0


def drop_synthetic_flat_runs(
    bars: Sequence[CanonicalOHLC],
    *,
    min_run_minutes: int = DEFAULT_MIN_RUN_MINUTES,
    min_run_bars: int = DEFAULT_MIN_RUN_BARS,
) -> tuple[list[CanonicalOHLC], int]:
    """Return (bars without synthetic flat runs, number of removed bars).

    ``bars`` must be sorted by timestamp and free of duplicates.
    """
    records = list(bars)
    if len(records) < min_run_bars:
        return records, 0
    step = _base_step_seconds(records)
    needed_bars = max(min_run_bars, int(-(-min_run_minutes * 60 // step)))

    frozen: list[bool] = []
    contiguous: list[bool] = []
    previous = None
    for bar in records:
        is_flat = bar.open == bar.high == bar.low == bar.close
        joined = previous is not None and (bar.timestamp - previous.timestamp).total_seconds() == step
        repeats = previous is not None and bar.close == previous.close
        frozen.append(is_flat and (repeats or not joined))
        contiguous.append(joined)
        previous = bar

    remove = [False] * len(records)
    start = None
    for index in range(len(records) + 1):
        in_run = index < len(records) and frozen[index] and (start is None or contiguous[index])
        if in_run:
            if start is None:
                start = index
            continue
        if start is not None and index - start >= needed_bars:
            for position in range(start, index):
                remove[position] = True
        start = index if index < len(records) and frozen[index] else None

    cleaned = [bar for bar, drop in zip(records, remove) if not drop]
    return cleaned, len(records) - len(cleaned)


__all__ = ["DEFAULT_MIN_RUN_BARS", "DEFAULT_MIN_RUN_MINUTES", "drop_synthetic_flat_runs"]

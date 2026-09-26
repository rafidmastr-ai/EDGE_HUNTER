from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.research.resampling import resample_ohlc


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bar(index: int, open_: float, high: float, low: float, close: float) -> CanonicalOHLC:
    return CanonicalOHLC(
        timestamp=BASE + timedelta(minutes=index),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
    )


class ResamplingTests(unittest.TestCase):
    def test_five_minute_aggregation_uses_last_source_timestamp(self) -> None:
        bars = tuple(bar(i, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10))
        result = resample_ohlc(bars, "M5")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].timestamp, bars[4].timestamp)
        self.assertEqual(result[0].open, bars[0].open)
        self.assertEqual(result[0].high, max(b.high for b in bars[:5]))
        self.assertEqual(result[0].low, min(b.low for b in bars[:5]))
        self.assertEqual(result[0].close, bars[4].close)

    def test_future_bar_does_not_change_previous_resampled_candle(self) -> None:
        initial = tuple(bar(i, 100, 101 + i, 99 - i, 100 + i * 0.1) for i in range(7))
        before = resample_ohlc(initial, "M5")
        extended = initial + (bar(7, 101, 103, 100, 102),)
        after = resample_ohlc(extended, "M5")
        self.assertEqual(before[0], after[0])

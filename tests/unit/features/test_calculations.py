from __future__ import annotations

import unittest

from app.features.calculations import atr_wilder, ema, rsi_wilder
from app.data.schema import CanonicalOHLC
from datetime import datetime, timedelta, timezone
from decimal import Decimal


def _bars(closes: list[float]) -> list[CanonicalOHLC]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars: list[CanonicalOHLC] = []
    for index, close in enumerate(closes):
        value = Decimal(str(close))
        bars.append(
            CanonicalOHLC(
                timestamp=start + timedelta(minutes=index),
                open=value,
                high=value + Decimal("1"),
                low=value - Decimal("1"),
                close=value,
            )
        )
    return bars


class CalculationTests(unittest.TestCase):
    def test_ema_matches_known_period_three_sequence(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ema(values, 3)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[3], 3.0)
        self.assertAlmostEqual(result[4], 4.0)

    def test_rsi_rising_sequence_reaches_100(self) -> None:
        result = rsi_wilder([1, 2, 3, 4, 5, 6], 3)
        self.assertIsNone(result[2])
        self.assertAlmostEqual(result[3], 100.0)
        self.assertAlmostEqual(result[5], 100.0)

    def test_atr_uses_first_period_true_ranges_and_wilder_updates(self) -> None:
        bars = _bars([10, 11, 12, 13])
        result = atr_wilder(bars, 3)
        self.assertIsNone(result[1])
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[3], 2.0)

    def test_invalid_period_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ema([1.0, 2.0], 0)


if __name__ == "__main__":
    unittest.main()

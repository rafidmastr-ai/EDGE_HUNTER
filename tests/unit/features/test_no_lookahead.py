from __future__ import annotations

import unittest
from datetime import timedelta
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from .test_engine import make_bars


class NoLookaheadTests(unittest.TestCase):
    def test_future_bars_do_not_change_existing_feature_values(self) -> None:
        engine = FeatureEngine()
        base = make_bars(210)
        future: list[CanonicalOHLC] = []
        step = base[-1].timestamp - base[-2].timestamp
        previous = base[-1]
        for offset in range(1, 16):
            close = previous.close + Decimal("0.25")
            current = CanonicalOHLC(
                timestamp=previous.timestamp + step,
                open=previous.close,
                high=close + Decimal("0.5"),
                low=previous.close - Decimal("0.2"),
                close=close,
            )
            future.append(current)
            previous = current

        first = engine.compute(base, "XAUUSD", "M1")
        second = engine.compute(base + future, "XAUUSD", "M1")
        names = (
            "trend.ema_20",
            "trend.ema_50",
            "momentum.rsi",
            "volatility.atr",
            "momentum.macd_histogram",
        )
        for index in range(len(base)):
            for name in names:
                self.assertEqual(first.snapshots[index].get(name), second.snapshots[index].get(name))

    def test_unsorted_input_is_rejected_instead_of_silently_reordering_history(self) -> None:
        bars = make_bars(5)
        reordered = [bars[0], bars[1], bars[3], bars[2], bars[4]]
        with self.assertRaises(ValueError):
            FeatureEngine().compute(reordered, "XAUUSD", "M1")


if __name__ == "__main__":
    unittest.main()

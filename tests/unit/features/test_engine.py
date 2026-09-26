from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureConfig, FeatureEngine


def make_bars(count: int, *, with_volume: bool = False) -> list[CanonicalOHLC]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars: list[CanonicalOHLC] = []
    for index in range(count):
        close = Decimal("100") + Decimal(index) + Decimal(index % 3) / Decimal("10")
        open_ = close - Decimal("0.5")
        bars.append(
            CanonicalOHLC(
                timestamp=start + timedelta(minutes=index),
                open=open_,
                high=close + Decimal("0.75"),
                low=open_ - Decimal("0.25"),
                close=close,
                volume=Decimal(str(100 + index)) if with_volume else None,
            )
        )
    return bars


class FeatureEngineTests(unittest.TestCase):
    def test_engine_returns_canonical_series_and_metadata(self) -> None:
        engine = FeatureEngine()
        series = engine.compute(make_bars(210), "xauusd", "M1")
        self.assertEqual(series.symbol, "XAUUSD")
        self.assertEqual(series.timeframe, "M1")
        self.assertEqual(len(series.snapshots), 210)
        self.assertEqual(series.engine_version, "phase03-v1")
        self.assertFalse(series.volume_features_enabled)
        self.assertTrue(series.snapshots[-1].warmup_complete)
        self.assertIn("momentum.rsi", series.snapshots[-1].values)
        self.assertIn("volatility.atr_pct", series.snapshots[-1].values)

    def test_parameterization_changes_warmup_and_output(self) -> None:
        config = FeatureConfig(ema_periods=(5, 10), rsi_period=5, atr_period=5,
                               macd_fast_period=3, macd_slow_period=6,
                               macd_signal_period=3, volatility_period=5,
                               volume_period=5, return_periods=(1, 3))
        series = FeatureEngine(config).compute(make_bars(20), "GBPUSD", "M5")
        self.assertEqual(series.warmup_bars_required, 10)
        self.assertFalse(series.snapshots[8].warmup_complete)
        self.assertTrue(series.snapshots[9].warmup_complete)
        self.assertEqual(series.snapshots[0].get("momentum.rsi"), None)

    def test_valid_volume_enables_volume_features_only_when_all_rows_are_valid(self) -> None:
        bars = make_bars(25, with_volume=True)
        series = FeatureEngine().compute(bars, "XAUUSD", "M15")
        self.assertTrue(series.volume_features_enabled)
        self.assertEqual(series.snapshots[0].get("volume.value"), 100.0)
        self.assertIsNone(series.snapshots[0].get("volume.ratio"))
        broken = list(bars)
        broken[7] = CanonicalOHLC(
            timestamp=broken[7].timestamp,
            open=broken[7].open,
            high=broken[7].high,
            low=broken[7].low,
            close=broken[7].close,
            volume=None,
        )
        broken_series = FeatureEngine().compute(broken, "XAUUSD", "M15")
        self.assertFalse(broken_series.volume_features_enabled)
        self.assertIsNone(broken_series.snapshots[-1].get("volume.value"))

    def test_unsorted_or_duplicate_timestamps_are_rejected(self) -> None:
        bars = make_bars(4)
        duplicate = list(bars)
        duplicate[2] = duplicate[1]
        with self.assertRaises(ValueError):
            FeatureEngine().compute(duplicate, "XAUUSD", "M1")


if __name__ == "__main__":
    unittest.main()

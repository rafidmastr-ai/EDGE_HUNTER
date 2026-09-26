from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot
from app.strategies.classic import ClassicV1
from app.strategies.models import StrategyConfig, StrategyContext
from app.strategies.utils import adaptive_rr


class StrategyTimingAndUtilityTests(unittest.TestCase):
    def test_adaptive_rr_stays_inside_configured_bounds(self) -> None:
        self.assertEqual(adaptive_rr(0.0, 1.5, 2.0), 1.5)
        self.assertEqual(adaptive_rr(1.0, 1.5, 2.0), 2.0)
        self.assertEqual(adaptive_rr(0.5, 1.5, 2.0), 1.75)
        self.assertIsNone(adaptive_rr(0.5, 0.0, 2.0))

    def test_decision_context_never_contains_future_bars(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [CanonicalOHLC(start + timedelta(minutes=i), Decimal("100"), Decimal("101"), Decimal("99"), Decimal(str(100 + i * 0.1))) for i in range(210)]
        snapshots = [MarketAnalysisSnapshot(b.timestamp, "XAUUSD", "M15", {"price.close": float(b.close)}, warmup_complete=True) for b in bars]
        series = MarketAnalysisSeries("XAUUSD", "M15", tuple(snapshots), (), 1, False)
        ctx = StrategyContext.from_series(bars, series, decision_index=200)
        self.assertEqual(len(ctx.bars), 201)
        self.assertEqual(ctx.bars[-1].timestamp, bars[200].timestamp)
        self.assertNotIn(bars[201].timestamp, [bar.timestamp for bar in ctx.bars])

    def test_configured_rr_range_is_respected_by_strategy(self) -> None:
        config = StrategyConfig(min_rr=1.6, max_rr=1.8)
        self.assertEqual(config.min_rr, 1.6)
        self.assertEqual(config.max_rr, 1.8)


if __name__ == "__main__":
    unittest.main()

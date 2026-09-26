from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot
from app.strategies.models import SignalDirection, SignalState, StrategyContext
from app.strategies.smc import SMCV1


def make_smc_context() -> StrategyContext:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        CanonicalOHLC(start + timedelta(minutes=i), Decimal("100"), Decimal("102"), Decimal("96"), Decimal("99"))
        for i in range(200)
    ]
    bars[-2] = CanonicalOHLC(start + timedelta(minutes=198), Decimal("100"), Decimal("104"), Decimal("95"), Decimal("100"))
    bars[-1] = CanonicalOHLC(start + timedelta(minutes=199), Decimal("98"), Decimal("106"), Decimal("94"), Decimal("105"))
    snapshots = []
    for bar in bars:
        snapshots.append(MarketAnalysisSnapshot(
            timestamp=bar.timestamp, symbol="XAUUSD", timeframe="M15",
            values={"structure.body_ratio": abs(float(bar.close) - float(bar.open)) / (float(bar.high) - float(bar.low))},
            warmup_complete=True,
        ))
    series = MarketAnalysisSeries("XAUUSD", "M15", tuple(snapshots), (), 1, False)
    return StrategyContext.from_series(bars, series)


class SMCStrategyTests(unittest.TestCase):
    def test_liquidity_sweep_displacement_and_bos_generate_buy(self) -> None:
        signal = SMCV1().generate(make_smc_context())
        self.assertEqual(signal.state, SignalState.SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.BUY)
        self.assertLess(signal.stop_loss, signal.entry)
        self.assertGreater(signal.target, signal.entry)
        self.assertGreaterEqual(signal.risk_reward, 1.5)
        self.assertLessEqual(signal.risk_reward, 2.0)
        self.assertIn("liquidity sweep", signal.evidence)
        self.assertIn("local structure break", signal.evidence)

    def test_without_sweep_returns_no_signal(self) -> None:
        context = make_smc_context()
        bars = list(context.bars)
        bars[-1] = CanonicalOHLC(
            bars[-1].timestamp, Decimal("100"), Decimal("106"), Decimal("99"), Decimal("105")
        )
        snapshots = list(context.analysis.snapshots)
        snapshots[-1] = MarketAnalysisSnapshot(
            timestamp=bars[-1].timestamp, symbol="XAUUSD", timeframe="M15",
            values={"structure.body_ratio": 5/7}, warmup_complete=True
        )
        series = MarketAnalysisSeries("XAUUSD", "M15", tuple(snapshots), (), 1, False)
        signal = SMCV1().generate(StrategyContext.from_series(bars, series))
        self.assertEqual(signal.state, SignalState.NO_SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.NO_SIGNAL)


if __name__ == "__main__":
    unittest.main()

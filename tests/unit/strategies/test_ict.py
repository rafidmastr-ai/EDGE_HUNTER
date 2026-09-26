from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot
from app.strategies.ict import ICTV1
from app.strategies.models import SignalDirection, SignalState, StrategyContext


def make_ict_context(*, bullish: bool = True) -> StrategyContext:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        CanonicalOHLC(start + timedelta(minutes=i), Decimal("100"), Decimal("102"), Decimal("98"), Decimal("100"))
        for i in range(200)
    ]
    if bullish:
        bars[-3] = CanonicalOHLC(start + timedelta(minutes=197), Decimal("100"), Decimal("102"), Decimal("99"), Decimal("101"))
        bars[-2] = CanonicalOHLC(start + timedelta(minutes=198), Decimal("101"), Decimal("104"), Decimal("100"), Decimal("102"))
        bars[-1] = CanonicalOHLC(start + timedelta(minutes=199), Decimal("103"), Decimal("108"), Decimal("103"), Decimal("107"))
        ema = 105.0
    else:
        bars[-3] = CanonicalOHLC(start + timedelta(minutes=197), Decimal("100"), Decimal("102"), Decimal("98"), Decimal("100"))
        bars[-2] = CanonicalOHLC(start + timedelta(minutes=198), Decimal("99"), Decimal("101"), Decimal("96"), Decimal("98"))
        bars[-1] = CanonicalOHLC(start + timedelta(minutes=199), Decimal("95"), Decimal("95"), Decimal("91"), Decimal("92"))
        ema = 94.0
    snaps = []
    for i, bar in enumerate(bars):
        body_ratio = abs(float(bar.close) - float(bar.open)) / max(float(bar.high) - float(bar.low), 1e-9)
        values = {"structure.body_ratio": body_ratio, "trend.ema_20": ema}
        snaps.append(MarketAnalysisSnapshot(bar.timestamp, "XAUUSD", "M15", values, warmup_complete=True))
    series = MarketAnalysisSeries("XAUUSD", "M15", tuple(snaps), (), 1, False)
    return StrategyContext.from_series(bars, series)


class ICTStrategyTests(unittest.TestCase):
    def test_bullish_fvg_and_displacement_generate_buy(self) -> None:
        signal = ICTV1().generate(make_ict_context())
        self.assertEqual(signal.state, SignalState.SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.BUY)
        self.assertLess(signal.stop_loss, signal.entry)
        self.assertGreater(signal.target, signal.entry)
        self.assertIn("three-candle fair-value gap", signal.evidence)

    def test_bearish_fvg_and_displacement_generate_sell(self) -> None:
        signal = ICTV1().generate(make_ict_context(bullish=False))
        self.assertEqual(signal.state, SignalState.SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.SELL)
        self.assertGreater(signal.stop_loss, signal.entry)
        self.assertLess(signal.target, signal.entry)

    def test_missing_warmup_is_not_traded(self) -> None:
        context = make_ict_context()
        snapshots = list(context.analysis.snapshots)
        snapshots[-1] = MarketAnalysisSnapshot(
            snapshots[-1].timestamp, "XAUUSD", "M15",
            snapshots[-1].values, warmup_complete=False
        )
        series = MarketAnalysisSeries("XAUUSD", "M15", tuple(snapshots), (), 200, False)
        signal = ICTV1().generate(StrategyContext.from_series(context.bars, series))
        self.assertEqual(signal.state, SignalState.INSUFFICIENT_DATA)


if __name__ == "__main__":
    unittest.main()

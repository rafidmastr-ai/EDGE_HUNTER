from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot
from app.strategies.models import SignalDirection, SignalState, StrategyConfig, StrategyContext, StrategySignal


def make_context(index: int = 5, *, warmup: bool = True) -> StrategyContext:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = tuple(
        CanonicalOHLC(
            timestamp=start + timedelta(minutes=i),
            open=Decimal("100") + Decimal(i),
            high=Decimal("101") + Decimal(i),
            low=Decimal("99") + Decimal(i),
            close=Decimal("100.5") + Decimal(i),
        )
        for i in range(index + 1)
    )
    snapshots = tuple(
        MarketAnalysisSnapshot(
            timestamp=bar.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            values={"price.close": float(bar.close)},
            warmup_complete=warmup,
        )
        for bar in bars
    )
    series = MarketAnalysisSeries(
        symbol="XAUUSD",
        timeframe="M15",
        snapshots=snapshots,
        definitions=(),
        warmup_bars_required=1,
        volume_features_enabled=False,
    )
    return StrategyContext.from_series(bars, series, decision_index=index)


class StrategyModelTests(unittest.TestCase):
    def test_context_is_sliced_at_decision_time(self) -> None:
        context = make_context(5)
        self.assertEqual(context.decision_index, 5)
        self.assertEqual(len(context.bars), 6)
        self.assertEqual(len(context.analysis.snapshots), 6)
        self.assertEqual(context.current.timestamp, context.bars[-1].timestamp)

    def test_context_rejects_length_mismatch(self) -> None:
        context = make_context(5)
        with self.assertRaises(ValueError):
            StrategyContext.from_series(context.bars[:-1], context.analysis)

    def test_context_rejects_bad_index(self) -> None:
        context = make_context(5)
        with self.assertRaises(IndexError):
            StrategyContext.from_series(context.bars, context.analysis, decision_index=99)

    def test_config_validates_rr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            StrategyConfig(min_rr=2.0, max_rr=1.5)

    def test_signal_requires_one_target_for_signal_state(self) -> None:
        context = make_context(5)
        signal = StrategySignal(
            timestamp=context.current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=SignalDirection.BUY,
            state=SignalState.SIGNAL,
            entry=101.0,
            stop_loss=100.0,
            target=103.0,
            risk_reward=2.0,
            entry_logic="test",
            invalidation="test",
            stop_loss_logic="test",
            target_logic="test",
            evidence=("test",),
            strategy_name="Test",
            variant="Test_V1",
        )
        self.assertEqual(signal.direction, SignalDirection.BUY)
        self.assertIsNotNone(signal.target)

    def test_non_signal_cannot_expose_trade_prices(self) -> None:
        context = make_context(5)
        with self.assertRaises(ValueError):
            StrategySignal(
                timestamp=context.current.timestamp,
                symbol=context.symbol,
                timeframe=context.timeframe,
                direction=SignalDirection.NO_SIGNAL,
                state=SignalState.NO_SIGNAL,
                entry=101.0,
                stop_loss=None,
                target=None,
                risk_reward=None,
                entry_logic="none",
                invalidation="none",
                stop_loss_logic="none",
                target_logic="none",
                strategy_name="Test",
                variant="Test_V1",
            )


if __name__ == "__main__":
    unittest.main()

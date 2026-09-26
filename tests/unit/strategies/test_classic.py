from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.models import FeatureDefinition, MarketAnalysisSeries, MarketAnalysisSnapshot
from app.strategies.classic import ClassicV1
from app.strategies.models import SignalDirection, SignalState, StrategyConfig, StrategyContext


def build_classic_context(*, bullish: bool = True, body_ratio: float = 0.8, future_shift: float = 0.0) -> StrategyContext:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    snapshots = []
    for i in range(205):
        base = 100 + i * 0.05
        bars.append(
            CanonicalOHLC(
                timestamp=start + timedelta(minutes=i),
                open=Decimal(str(base)),
                high=Decimal(str(base + 0.3)),
                low=Decimal(str(base - 0.2)),
                close=Decimal(str(base + 0.1)),
            )
        )
    # A bullish/bearish breakout bar. Its preceding five-bar range stays below/above it.
    if bullish:
        entry, open_, high, low = 112.0, 111.0, 113.0, 110.8
        ema_values = {"trend.ema_20": 109.0, "trend.ema_50": 107.0, "trend.ema_200": 102.0}
        rsi, hist = 62.0, 0.5
    else:
        entry, open_, high, low = 90.0, 91.0, 91.2, 88.0
        ema_values = {"trend.ema_20": 93.0, "trend.ema_50": 95.0, "trend.ema_200": 101.0}
        rsi, hist = 38.0, -0.5
    bars[-1] = CanonicalOHLC(
        timestamp=start + timedelta(minutes=204),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(entry + future_shift * 0)),
    )
    for i, bar in enumerate(bars[:-1]):
        snapshots.append(MarketAnalysisSnapshot(
            timestamp=bar.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            values={
                "price.close": float(bar.close),
                "trend.ema_20": 105.0,
                "trend.ema_50": 104.0,
                "trend.ema_200": 103.0,
                "momentum.rsi": 55.0,
                "momentum.macd_histogram": 0.1,
                "structure.body_ratio": 0.7,
            },
            warmup_complete=True,
        ))
    snap_values = {
        "price.close": entry,
        **ema_values,
        "momentum.rsi": rsi,
        "momentum.macd_histogram": hist,
        "structure.body_ratio": body_ratio,
    }
    snapshots.append(MarketAnalysisSnapshot(
        timestamp=bars[-1].timestamp,
        symbol="XAUUSD",
        timeframe="M15",
        values=snap_values,
        warmup_complete=True,
    ))
    series = MarketAnalysisSeries(
        symbol="XAUUSD", timeframe="M15", snapshots=tuple(snapshots),
        definitions=(FeatureDefinition("price.close", "price", "close", "OHLC", 1),),
        warmup_bars_required=200, volume_features_enabled=False,
    )
    return StrategyContext.from_series(bars, series)


class ClassicStrategyTests(unittest.TestCase):
    def test_bullish_signal_has_single_target_and_valid_geometry(self) -> None:
        signal = ClassicV1().generate(build_classic_context())
        self.assertEqual(signal.state, SignalState.SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.BUY)
        self.assertLess(signal.stop_loss, signal.entry)
        self.assertGreater(signal.target, signal.entry)
        self.assertGreaterEqual(signal.risk_reward, 1.5)
        self.assertLessEqual(signal.risk_reward, 2.0)
        self.assertEqual(len([signal.target]), 1)

    def test_bearish_signal(self) -> None:
        signal = ClassicV1().generate(build_classic_context(bullish=False))
        self.assertEqual(signal.state, SignalState.SIGNAL)
        self.assertEqual(signal.direction, SignalDirection.SELL)
        self.assertGreater(signal.stop_loss, signal.entry)
        self.assertLess(signal.target, signal.entry)

    def test_insufficient_warmup_returns_explicit_state(self) -> None:
        context = build_classic_context()
        short_snap = tuple(
            s.__class__(s.timestamp, s.symbol, s.timeframe, s.values, s.metadata, False)
            for s in context.analysis.snapshots
        )
        short_series = MarketAnalysisSeries(
            context.analysis.symbol, context.analysis.timeframe, short_snap,
            context.analysis.definitions, context.analysis.warmup_bars_required,
            context.analysis.volume_features_enabled,
        )
        short_context = StrategyContext.from_series(context.bars, short_series)
        signal = ClassicV1().generate(short_context)
        self.assertEqual(signal.state, SignalState.INSUFFICIENT_DATA)
        self.assertEqual(signal.direction, SignalDirection.NO_SIGNAL)

    def test_rr_is_dynamic_with_setup_strength(self) -> None:
        weak = ClassicV1().generate(build_classic_context(body_ratio=0.56))
        strong = ClassicV1().generate(build_classic_context(body_ratio=0.95))
        self.assertEqual(weak.state, SignalState.SIGNAL)
        self.assertEqual(strong.state, SignalState.SIGNAL)
        self.assertLess(weak.risk_reward, strong.risk_reward)
        self.assertGreaterEqual(weak.risk_reward, 1.5)
        self.assertLessEqual(strong.risk_reward, 2.0)


if __name__ == "__main__":
    unittest.main()

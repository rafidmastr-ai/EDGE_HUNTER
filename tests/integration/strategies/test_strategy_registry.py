from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureConfig, FeatureEngine
from app.strategies import StrategyRegistry, StrategyConfig, StrategyContext


class StrategyRegistryIntegrationTests(unittest.TestCase):
    def test_all_initial_strategies_share_one_interface(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = []
        for i in range(210):
            close = Decimal("100") + Decimal(i) / Decimal("10")
            bars.append(CanonicalOHLC(
                timestamp=start + timedelta(minutes=i),
                open=close - Decimal("0.1"),
                high=close + Decimal("0.2"),
                low=close - Decimal("0.3"),
                close=close,
            ))
        feature_config = FeatureConfig(
            ema_periods=(20, 50, 200), rsi_period=14, atr_period=14,
            macd_fast_period=12, macd_slow_period=26, macd_signal_period=9,
            volatility_period=20, volume_period=20, return_periods=(1, 5, 20)
        )
        analysis = FeatureEngine(feature_config).compute(bars, "XAUUSD", "M15")
        context = StrategyContext.from_series(bars, analysis)
        registry = StrategyRegistry.default(StrategyConfig())
        signals = registry.evaluate_all(context)
        self.assertEqual(registry.variants(), ("Classic_V1", "SMC_V1", "ICT_V1"))
        self.assertEqual(len(signals), 3)
        for signal in signals:
            self.assertEqual(signal.timestamp, context.current.timestamp)
            self.assertEqual(signal.symbol, "XAUUSD")
            self.assertIn(signal.variant, registry.variants())

    def test_evaluate_map_has_unique_strategy_names(self) -> None:
        registry = StrategyRegistry.default()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(len(set(registry.names())), len(registry.names()))
        self.assertTrue(all(name for name in registry.names()))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.features.engine import FeatureEngine
from app.data.schema import CanonicalOHLC
from app.signals.engine import SignalConfidenceEngine
from app.strategies.models import StrategyContext
from app.strategies.registry import StrategyRegistry


class SignalEngineIntegrationTests(unittest.TestCase):
    def test_initial_strategies_run_through_common_signal_engine(self) -> None:
        bars = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(220):
            base = 100.0 + index * 0.01
            bars.append(
                CanonicalOHLC(
                    timestamp=start + timedelta(minutes=index),
                    open=Decimal(str(base)),
                    high=Decimal(str(base + 0.05)),
                    low=Decimal(str(base - 0.05)),
                    close=Decimal(str(base + 0.02)),
                )
            )
        analysis = FeatureEngine().compute(bars, "XAUUSD", "M1")
        context = StrategyContext.from_series(bars, analysis)
        decision = SignalConfidenceEngine(StrategyRegistry.default()).analyze(context)
        self.assertEqual(decision.timestamp, context.current.timestamp)
        self.assertEqual(decision.symbol, "XAUUSD")
        self.assertEqual(len(decision.strategy_evaluations), 3)
        self.assertEqual(decision.metadata["weak_signals_visible"], True)

    def test_api_ready_output_is_json_like(self) -> None:
        bars = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(220):
            value = 200.0 + index * 0.01
            bars.append(
                CanonicalOHLC(
                    timestamp=start + timedelta(minutes=index),
                    open=Decimal(str(value)),
                    high=Decimal(str(value + 0.1)),
                    low=Decimal(str(value - 0.1)),
                    close=Decimal(str(value + 0.03)),
                )
            )
        analysis = FeatureEngine().compute(bars, "GBPUSD", "M1")
        context = StrategyContext.from_series(bars, analysis)
        decision = SignalConfidenceEngine(StrategyRegistry.default()).analyze(context)
        payload = decision.to_dict()
        self.assertIn("direction", payload)
        self.assertIn("confidence", payload)
        self.assertIn("strategy_evaluations", payload)

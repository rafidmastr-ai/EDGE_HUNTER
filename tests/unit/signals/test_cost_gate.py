from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.signals.costs import CostGate, CostModel, split_symbol
from app.signals.engine import SignalConfidenceEngine
from app.strategies.models import SignalDirection, SignalState, StrategySignal
from app.strategies.registry import StrategyRegistry

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def setup(symbol: str, entry: float, stop: float, *, name: str = "Classic", direction=SignalDirection.BUY, metadata=None) -> StrategySignal:
    risk = abs(entry - stop)
    target = entry + 2 * risk if direction == SignalDirection.BUY else entry - 2 * risk
    return StrategySignal(
        timestamp=TS, symbol=symbol, timeframe="M15", direction=direction, state=SignalState.SIGNAL,
        entry=entry, stop_loss=stop, target=target, risk_reward=2.0,
        entry_logic="e", invalidation="i", stop_loss_logic="s", target_logic="t", evidence=("x",),
        strategy_name=name, variant=f"{name}_V1", metadata=metadata or {},
    )


class Stub:
    def __init__(self, signal):
        self.signal, self.name, self.variant = signal, signal.strategy_name, signal.variant

    def generate(self, context):
        return self.signal


class CostModelTests(unittest.TestCase):
    def test_forex_uses_pips_and_jpy_pip_size(self) -> None:
        model = CostModel()
        self.assertAlmostEqual(model.estimate("EURUSD", 1.1).value, (0.8 + 0.3 + 0.7) * 0.0001)
        self.assertAlmostEqual(model.estimate("USD/JPY", 150.0).value, (0.9 + 0.3 + 0.7) * 0.01)
        self.assertAlmostEqual(model.estimate("EURNOK", 11.0).value, (2.5 + 0.3 + 0.7) * 0.0001)

    def test_metal_crypto_and_other(self) -> None:
        model = CostModel()
        self.assertEqual(model.estimate("XAUUSD", 3000.0).value, 0.42)
        self.assertAlmostEqual(model.estimate("BTC/USD", 60000.0).value, 60.0)
        self.assertAlmostEqual(model.estimate("SPX500", 5000.0).value, 2.5)

    def test_split_symbol(self) -> None:
        self.assertEqual(split_symbol("eur/usd"), ("EUR", "USD"))
        self.assertEqual(split_symbol("XAUUSD"), ("XAU", "USD"))


class CostGateTests(unittest.TestCase):
    def test_tight_stop_is_withheld_with_reason_and_no_prices(self) -> None:
        # EURUSD cost 1.8 pips; 5-pip stop -> cost_r 0.36 > 0.10
        out = CostGate().apply([setup("EURUSD", 1.10000, 1.09950)])[0]
        self.assertEqual(out.state, SignalState.NO_SIGNAL)
        self.assertIsNone(out.entry)
        self.assertEqual(out.metadata["cost_decision"], "rejected")
        self.assertEqual(out.metadata["original_direction"], "BUY")
        self.assertAlmostEqual(out.metadata["cost_r"], 0.36, places=4)
        self.assertTrue(out.evidence[0].startswith("cost_gate_rejected"))

    def test_wide_stop_is_accepted_and_annotated_unchanged_prices(self) -> None:
        signal = setup("EURUSD", 1.10000, 1.09700)  # 30 pips -> cost_r 0.06
        out = CostGate().apply([signal])[0]
        self.assertEqual(out.state, SignalState.SIGNAL)
        self.assertEqual((out.entry, out.stop_loss, out.target, out.risk_reward), (signal.entry, signal.stop_loss, signal.target, signal.risk_reward))
        self.assertEqual(out.metadata["cost_decision"], "accepted")
        self.assertAlmostEqual(out.metadata["cost_r"], 0.06, places=4)
        self.assertAlmostEqual(out.metadata["net_risk_reward_after_cost"], 1.94, places=4)
        self.assertAlmostEqual(out.metadata["min_stop_distance"], 0.0018, places=8)

    def test_boundary_is_inclusive_and_threshold_configurable(self) -> None:
        signal = setup("XAUUSD", 3000.0, 2995.8)  # cost 0.42 / risk 4.2 = 0.10
        self.assertEqual(CostGate(max_cost_r=0.10).apply([signal])[0].state, SignalState.SIGNAL)
        self.assertEqual(CostGate(max_cost_r=0.05).apply([signal])[0].state, SignalState.NO_SIGNAL)

    def test_non_signals_pass_through_and_ml_annotations_are_kept(self) -> None:
        withheld = setup("EURUSD", 1.1, 1.0999, metadata={"ml_probability": 0.6, "other": 1})
        out = CostGate().apply([withheld])[0]
        self.assertEqual(out.metadata["ml_probability"], 0.6)
        self.assertNotIn("other", out.metadata)
        self.assertIs(CostGate().apply([out])[0], out)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CostGate(max_cost_r=0.0)


class EngineIntegrationTests(unittest.TestCase):
    def test_engine_applies_gate_and_keeps_raw_outputs(self) -> None:
        from app.features.engine import FeatureEngine
        from app.data.schema import CanonicalOHLC
        from app.strategies.models import StrategyContext
        from datetime import timedelta
        from decimal import Decimal

        bars = [CanonicalOHLC(timestamp=TS + timedelta(minutes=15 * i), open=Decimal("1.1"), high=Decimal("1.1005"), low=Decimal("1.0995"), close=Decimal("1.1"), volume=None) for i in range(30)]
        analysis = FeatureEngine().compute(bars, "EURUSD", "M15")
        context = StrategyContext.from_series(bars, analysis)
        ts = context.current.timestamp
        tight = StrategySignal(**{**setup("EURUSD", 1.1, 1.0995).__dict__, "timestamp": ts})
        wide = StrategySignal(**{**setup("EURUSD", 1.1, 1.097, name="SMC").__dict__, "timestamp": ts})
        engine = SignalConfidenceEngine(StrategyRegistry((Stub(tight), Stub(wide))), cost_gate=CostGate())
        raw, final = engine.evaluate_signals(context)
        self.assertEqual([s.state for s in raw], [SignalState.SIGNAL, SignalState.SIGNAL])
        by_name = {s.strategy_name: s for s in final}
        self.assertEqual(by_name["Classic"].state, SignalState.NO_SIGNAL)
        self.assertEqual(by_name["SMC"].state, SignalState.SIGNAL)
        without = SignalConfidenceEngine(StrategyRegistry((Stub(tight), Stub(wide))))
        self.assertEqual(without.evaluate_signals(context)[1], without.evaluate_signals(context)[0])


if __name__ == "__main__":
    unittest.main()

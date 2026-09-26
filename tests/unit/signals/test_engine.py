from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.signals.config import ConfidenceConfig
from app.signals.engine import SignalConfidenceEngine
from app.strategies.models import SignalDirection, SignalState, StrategySignal
from app.strategies.registry import StrategyRegistry


TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_signal(
    name: str,
    direction: SignalDirection,
    confidence_quality: float = 1.0,
    metadata=None,
) -> StrategySignal:
    if direction == SignalDirection.NO_SIGNAL:
        return StrategySignal(
            timestamp=TS,
            symbol="XAUUSD",
            timeframe="M15",
            direction=SignalDirection.NO_SIGNAL,
            state=SignalState.NO_SIGNAL,
            entry=None,
            stop_loss=None,
            target=None,
            risk_reward=None,
            entry_logic="none",
            invalidation="none",
            stop_loss_logic="none",
            target_logic="none",
            evidence=("no setup",),
            strategy_name=name,
            variant=f"{name}_V1",
            metadata=metadata or {},
        )
    return StrategySignal(
        timestamp=TS,
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        state=SignalState.SIGNAL,
        entry=100.0,
        stop_loss=99.0 if direction == SignalDirection.BUY else 101.0,
        target=102.0 if direction == SignalDirection.BUY else 98.0,
        risk_reward=2.0,
        entry_logic="entry",
        invalidation="invalidation",
        stop_loss_logic="stop",
        target_logic="target",
        evidence=("evidence",),
        score_inputs={"quality_a": confidence_quality, "quality_b": confidence_quality},
        strategy_name=name,
        variant=f"{name}_V1",
        metadata=metadata or {},
    )


class StubStrategy:
    def __init__(self, signal: StrategySignal):
        self.signal = signal
        self.name = signal.strategy_name
        self.variant = signal.variant

    def generate(self, context):
        self.assert_context = (context.decision_index, len(context.bars))
        return self.signal


def engine_for(*signals: StrategySignal) -> SignalConfidenceEngine:
    return SignalConfidenceEngine(
        registry=StrategyRegistry(tuple(StubStrategy(item) for item in signals)),
        config=ConfidenceConfig(),
    )


class EngineTests(unittest.TestCase):
    def test_identical_inputs_produce_identical_confidence(self) -> None:
        signals = [make_signal("A", SignalDirection.BUY), make_signal("B", SignalDirection.BUY)]
        first = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15").to_dict()
        second = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15").to_dict()
        self.assertEqual(first, second)

    def test_weak_signal_is_visible_but_not_selected(self) -> None:
        signals = [make_signal("A", SignalDirection.BUY, confidence_quality=0.1)]
        decision = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15")
        self.assertEqual(decision.direction.value, "NO_CLEAR_SIGNAL")
        self.assertEqual(len(decision.strategy_evaluations), 1)
        self.assertTrue(decision.metadata["weak_signals_visible"])

    def test_conflicting_strategies_return_no_clear_signal(self) -> None:
        signals = [make_signal("A", SignalDirection.BUY), make_signal("B", SignalDirection.SELL)]
        decision = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15")
        self.assertEqual(decision.direction.value, "NO_CLEAR_SIGNAL")
        self.assertIsNone(decision.target)

    def test_no_signal_returns_zero_confidence_and_no_prices(self) -> None:
        signals = [make_signal("A", SignalDirection.NO_SIGNAL, confidence_quality=0.0)]
        decision = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15")
        self.assertEqual(decision.direction.value, "NO_CLEAR_SIGNAL")
        self.assertEqual(decision.confidence, 0.0)
        self.assertIsNone(decision.entry)
        self.assertIsNone(decision.target)

    def test_one_final_target_is_taken_from_one_representative_strategy(self) -> None:
        signals = [
            make_signal("A", SignalDirection.BUY, confidence_quality=0.7),
            make_signal("B", SignalDirection.BUY, confidence_quality=1.0),
        ]
        decision = engine_for(*signals).aggregate(signals, TS, "XAUUSD", "M15")
        self.assertEqual(decision.direction.value, "BUY")
        self.assertTrue(decision.metadata["one_final_target"])
        self.assertEqual(decision.selected_strategy, "B")
        self.assertIsNotNone(decision.target)

    def test_future_data_is_not_part_of_aggregation(self) -> None:
        signal = make_signal("A", SignalDirection.BUY, metadata={"future_score": 1.0})
        baseline = engine_for(signal).aggregate([signal], TS, "XAUUSD", "M15").to_dict()
        altered = make_signal("A", SignalDirection.BUY, metadata={"future_score": 0.0})
        comparison = engine_for(altered).aggregate([altered], TS, "XAUUSD", "M15").to_dict()
        self.assertEqual(baseline["confidence"], comparison["confidence"])

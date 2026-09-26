from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.signals.config import ConfidenceConfig
from app.signals.selector import select_direction
from app.strategies.models import SignalDirection, SignalState, StrategySignal


TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def signal(name: str, direction: SignalDirection, state: SignalState = SignalState.SIGNAL) -> StrategySignal:
    if state == SignalState.SIGNAL:
        return StrategySignal(
            timestamp=TS,
            symbol="XAUUSD",
            timeframe="M15",
            direction=direction,
            state=state,
            entry=100.0,
            stop_loss=99.0 if direction == SignalDirection.BUY else 101.0,
            target=102.0 if direction == SignalDirection.BUY else 98.0,
            risk_reward=2.0,
            entry_logic="entry",
            invalidation="invalidation",
            stop_loss_logic="stop",
            target_logic="target",
            evidence=("e",),
            score_inputs={"a": 1.0},
            strategy_name=name,
            variant=f"{name}_V1",
        )
    return StrategySignal(
        timestamp=TS,
        symbol="XAUUSD",
        timeframe="M15",
        direction=SignalDirection.NO_SIGNAL,
        state=state,
        entry=None,
        stop_loss=None,
        target=None,
        risk_reward=None,
        entry_logic="none",
        invalidation="none",
        stop_loss_logic="none",
        target_logic="none",
        evidence=("none",),
        strategy_name=name,
        variant=f"{name}_V1",
    )


class SelectorTests(unittest.TestCase):
    def test_two_buy_one_sell_selects_buy_under_default_gate(self) -> None:
        signals = [signal("A", SignalDirection.BUY), signal("B", SignalDirection.BUY), signal("C", SignalDirection.SELL)]
        result = select_direction(signals, [80.0, 80.0, 80.0])
        self.assertEqual(result.direction, SignalDirection.BUY)

    def test_equal_conflict_returns_no_signal(self) -> None:
        signals = [signal("A", SignalDirection.BUY), signal("B", SignalDirection.SELL)]
        result = select_direction(signals, [80.0, 80.0])
        self.assertEqual(result.direction, SignalDirection.NO_SIGNAL)

    def test_no_actionable_signals_returns_no_signal(self) -> None:
        signals = [signal("A", SignalDirection.NO_SIGNAL, SignalState.NO_SIGNAL)]
        result = select_direction(signals, [0.0])
        self.assertEqual(result.direction, SignalDirection.NO_SIGNAL)

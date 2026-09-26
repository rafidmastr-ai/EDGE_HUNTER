from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.backtest import BacktestConfig, BacktestEngine, TradeOutcome
from app.data.schema import CanonicalOHLC
from app.strategies.models import SignalDirection, SignalState, StrategySignal


class IntegrationStrategy:
    name = "Integration"
    variant = "Integration_V1"

    def generate(self, context):
        if context.decision_index == 0:
            return StrategySignal(
                timestamp=context.current.timestamp,
                symbol=context.symbol,
                timeframe=context.timeframe,
                direction=SignalDirection.BUY,
                state=SignalState.SIGNAL,
                entry=100.0,
                stop_loss=99.0,
                target=102.0,
                risk_reward=2.0,
                entry_logic="integration test",
                invalidation="integration test",
                stop_loss_logic="integration test",
                target_logic="integration test",
                evidence=("known synthetic outcome",),
                strategy_name=self.name,
                variant=self.variant,
            )
        return SignalSignalFactory.no_signal(context, self.name, self.variant)


class SignalSignalFactory:
    @staticmethod
    def no_signal(context, name: str, variant: str) -> StrategySignal:
        return StrategySignal(
            timestamp=context.current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
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
            evidence=("no signal",),
            strategy_name=name,
            variant=variant,
        )


def make_bars():
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    return (
        CanonicalOHLC(base, Decimal("100"), Decimal("100.2"), Decimal("99.8"), Decimal("100")),
        CanonicalOHLC(base + timedelta(minutes=1), Decimal("100"), Decimal("102.2"), Decimal("99.9"), Decimal("101.5")),
        CanonicalOHLC(base + timedelta(minutes=2), Decimal("101.5"), Decimal("101.8"), Decimal("101.2"), Decimal("101.6")),
    )


class BacktestIntegrationTests(unittest.TestCase):
    def test_strategy_engine_and_backtest_engine_share_the_same_signal_contract(self) -> None:
        result = BacktestEngine(BacktestConfig(starting_capital=10_000.0)).run(
            IntegrationStrategy(), make_bars(), symbol="XAUUSD", timeframe="M1"
        )
        self.assertEqual(result.strategy_name, "Integration")
        self.assertEqual(result.strategy_variant, "Integration_V1")
        self.assertEqual(result.metrics.total_trades, 1)
        self.assertEqual(result.trades[0].outcome, TradeOutcome.WIN)
        self.assertAlmostEqual(result.metrics.average_rr or 0.0, 2.0)

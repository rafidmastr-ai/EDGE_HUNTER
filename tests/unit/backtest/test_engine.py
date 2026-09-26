from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.backtest.config import BacktestConfig, IntrabarPolicy
from app.backtest.engine import BacktestEngine
from app.backtest.models import TradeOutcome
from app.data.schema import CanonicalOHLC
from app.strategies.models import SignalDirection, SignalState, StrategyContext, StrategySignal


BASE = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def bar(index: int, close: float, high: float | None = None, low: float | None = None) -> CanonicalOHLC:
    value = Decimal(str(close))
    return CanonicalOHLC(
        timestamp=BASE + timedelta(minutes=index),
        open=value,
        high=Decimal(str(high if high is not None else close + 0.5)),
        low=Decimal(str(low if low is not None else close - 0.5)),
        close=value,
    )


def signal_for(context: StrategyContext, direction: SignalDirection = SignalDirection.BUY) -> StrategySignal:
    entry = 100.0
    stop = 98.0 if direction == SignalDirection.BUY else 102.0
    target = 102.0 if direction == SignalDirection.BUY else 98.0
    return StrategySignal(
        timestamp=context.current.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        direction=direction,
        state=SignalState.SIGNAL,
        entry=entry,
        stop_loss=stop,
        target=target,
        risk_reward=1.0,
        entry_logic="synthetic close",
        invalidation="synthetic",
        stop_loss_logic="synthetic",
        target_logic="synthetic",
        evidence=("synthetic test evidence",),
        score_inputs={"test": 1.0},
        strategy_name="Synthetic",
        variant="Synthetic_V1",
    )


class ScheduledStrategy:
    name = "Synthetic"
    variant = "Synthetic_V1"

    def __init__(self, schedule: dict[int, SignalDirection], *, bad_timestamp: bool = False) -> None:
        self.schedule = schedule
        self.bad_timestamp = bad_timestamp
        self.seen_lengths: list[int] = []

    def generate(self, context: StrategyContext) -> StrategySignal:
        self.seen_lengths.append(len(context.bars))
        self.assert_no_future_data(context)
        direction = self.schedule.get(context.decision_index)
        if direction is None:
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
                evidence=("no setup",),
                strategy_name=self.name,
                variant=self.variant,
            )
        signal = signal_for(context, direction)
        if self.bad_timestamp:
            signal = StrategySignal(
                timestamp=context.current.timestamp + timedelta(minutes=1),
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                direction=signal.direction,
                state=signal.state,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                target=signal.target,
                risk_reward=signal.risk_reward,
                entry_logic=signal.entry_logic,
                invalidation=signal.invalidation,
                stop_loss_logic=signal.stop_loss_logic,
                target_logic=signal.target_logic,
                evidence=signal.evidence,
                score_inputs=signal.score_inputs,
                strategy_name=signal.strategy_name,
                variant=signal.variant,
            )
        return signal

    def assert_no_future_data(self, context: StrategyContext) -> None:
        if len(context.bars) != context.decision_index + 1:
            raise AssertionError("strategy context exposed future bars")
        if context.bars[-1].timestamp != context.current.timestamp:
            raise AssertionError("current snapshot is not the final available observation")


class BacktestEngineTests(unittest.TestCase):
    def test_long_target_closes_on_next_candle(self) -> None:
        bars = (
            bar(0, 100, high=103, low=97),
            bar(1, 101, high=103, low=99),
            bar(2, 101, high=102, low=100),
        )
        result = BacktestEngine().run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.metrics.wins, 1)
        self.assertEqual(result.trades[0].exit_reason, "TP")
        self.assertEqual(result.trades[0].entry_timestamp, bars[0].timestamp)
        self.assertEqual(result.trades[0].exit_timestamp, bars[1].timestamp)
        self.assertEqual(result.trades[0].bars_held, 1)

    def test_signal_candle_is_not_used_for_tp_or_sl(self) -> None:
        bars = (
            bar(0, 100, high=103, low=97),
            bar(1, 100, high=100.5, low=99.5),
        )
        result = BacktestEngine().run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.trades[0].outcome, TradeOutcome.EXPIRED)
        self.assertEqual(result.trades[0].exit_reason, "DATA_END")

    def test_stop_loss_closes_trade(self) -> None:
        bars = (
            bar(0, 100),
            bar(1, 99, high=100.2, low=97.0),
        )
        result = BacktestEngine().run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.metrics.losses, 1)
        self.assertEqual(result.trades[0].exit_reason, "SL")

    def test_intrabar_stop_first_is_default(self) -> None:
        bars = (bar(0, 100), bar(1, 100, high=103, low=97))
        result = BacktestEngine().run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.trades[0].outcome, TradeOutcome.LOSS)
        self.assertEqual(result.trades[0].exit_reason, "TP_AND_SL_STOP_FIRST")

    def test_intrabar_target_first_is_configurable(self) -> None:
        config = BacktestConfig(intrabar_policy=IntrabarPolicy.TARGET_FIRST)
        bars = (bar(0, 100), bar(1, 100, high=103, low=97))
        result = BacktestEngine(config).run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.trades[0].outcome, TradeOutcome.WIN)
        self.assertEqual(result.trades[0].exit_reason, "TP_AND_SL_TARGET_FIRST")

    def test_ambiguous_can_be_skipped_explicitly(self) -> None:
        config = BacktestConfig(intrabar_policy=IntrabarPolicy.SKIP_TRADE)
        bars = (bar(0, 100), bar(1, 100, high=103, low=97))
        result = BacktestEngine(config).run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.metrics.total_trades, 0)
        self.assertEqual(result.metrics.ambiguous_skipped, 1)

    def test_expiry_limit_is_applied_in_bars(self) -> None:
        config = BacktestConfig(max_bars_in_trade=2)
        bars = (bar(0, 100), bar(1, 100), bar(2, 100), bar(3, 100))
        result = BacktestEngine(config).run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.trades[0].outcome, TradeOutcome.EXPIRED)
        self.assertEqual(result.trades[0].exit_reason, "EXPIRY")
        self.assertEqual(result.trades[0].bars_held, 2)

    def test_short_target_is_supported(self) -> None:
        bars = (bar(0, 100), bar(1, 99, high=100.5, low=97.0))
        result = BacktestEngine().run(ScheduledStrategy({0: SignalDirection.SELL}), bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(result.metrics.wins, 1)
        self.assertEqual(result.trades[0].direction, "SELL")

    def test_spread_and_commission_are_applied_adversely(self) -> None:
        config = BacktestConfig(spread=0.2, commission_per_unit=0.1)
        bars = (bar(0, 100), bar(1, 101.5, high=103, low=99))
        result = BacktestEngine(config).run(ScheduledStrategy({0: SignalDirection.BUY}), bars, symbol="XAUUSD", timeframe="M1")
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_price, 100.1)
        self.assertAlmostEqual(trade.exit_price, 101.9)
        self.assertAlmostEqual(trade.pnl_net, 1.6)

    def test_signal_timestamp_must_match_decision_bar(self) -> None:
        bars = (bar(0, 100), bar(1, 100))
        with self.assertRaises(ValueError):
            BacktestEngine().run(
                ScheduledStrategy({0: SignalDirection.BUY}, bad_timestamp=True),
                bars,
                symbol="XAUUSD",
                timeframe="M1",
            )

    def test_strategy_context_is_sliced_to_decision_time(self) -> None:
        bars = tuple(bar(index, 100 + index * 0.01) for index in range(8))
        strategy = ScheduledStrategy({})
        BacktestEngine().run(strategy, bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(strategy.seen_lengths, list(range(1, 9)))

    def test_risk_sizing_uses_current_equity(self) -> None:
        config = BacktestConfig(starting_capital=10_000.0, risk_per_trade_pct=1.0)
        # First trade loses 100, second trade risks 1% of the reduced 9,900.
        bars = (
            bar(0, 100),
            bar(1, 99, high=100, low=97),
            bar(2, 100),
            bar(3, 102, high=103, low=99),
        )
        result = BacktestEngine(config).run(
            ScheduledStrategy({0: SignalDirection.BUY, 2: SignalDirection.BUY}),
            bars,
            symbol="XAUUSD",
            timeframe="M1",
        )
        self.assertEqual(len(result.trades), 2)
        self.assertAlmostEqual(result.trades[0].quantity, 50.0)
        self.assertAlmostEqual(result.trades[1].quantity, 49.5)


    def test_signal_symbol_must_match_run_symbol(self) -> None:
        class WrongSymbolStrategy(ScheduledStrategy):
            def generate(self, context: StrategyContext) -> StrategySignal:
                signal = super().generate(context)
                if signal.state == SignalState.SIGNAL:
                    return StrategySignal(
                        timestamp=signal.timestamp,
                        symbol="EURUSD",
                        timeframe=signal.timeframe,
                        direction=signal.direction,
                        state=signal.state,
                        entry=signal.entry,
                        stop_loss=signal.stop_loss,
                        target=signal.target,
                        risk_reward=signal.risk_reward,
                        entry_logic=signal.entry_logic,
                        invalidation=signal.invalidation,
                        stop_loss_logic=signal.stop_loss_logic,
                        target_logic=signal.target_logic,
                        evidence=signal.evidence,
                        score_inputs=signal.score_inputs,
                        strategy_name=signal.strategy_name,
                        variant=signal.variant,
                    )
                return signal

        with self.assertRaises(ValueError):
            BacktestEngine().run(
                WrongSymbolStrategy({0: SignalDirection.BUY}),
                (bar(0, 100), bar(1, 100)),
                symbol="XAUUSD",
                timeframe="M1",
            )

    def test_invalid_direction_geometry_is_rejected(self) -> None:
        class BadGeometryStrategy(ScheduledStrategy):
            def generate(self, context: StrategyContext) -> StrategySignal:
                signal = super().generate(context)
                if signal.state == SignalState.SIGNAL:
                    return signal.__class__(
                        timestamp=signal.timestamp,
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                        direction=signal.direction,
                        state=signal.state,
                        entry=100.0,
                        stop_loss=102.0,
                        target=98.0,
                        risk_reward=signal.risk_reward,
                        entry_logic=signal.entry_logic,
                        invalidation=signal.invalidation,
                        stop_loss_logic=signal.stop_loss_logic,
                        target_logic=signal.target_logic,
                        evidence=signal.evidence,
                        score_inputs=signal.score_inputs,
                        strategy_name=signal.strategy_name,
                        variant=signal.variant,
                    )
                return signal

        with self.assertRaises(ValueError):
            BacktestEngine().run(
                BadGeometryStrategy({0: SignalDirection.BUY}),
                (bar(0, 100), bar(1, 100)),
                symbol="XAUUSD",
                timeframe="M1",
            )

    def test_reproducibility_metadata_has_stable_data_fingerprint(self) -> None:
        bars = (bar(0, 100), bar(1, 101, high=103, low=99))
        strategy = ScheduledStrategy({0: SignalDirection.BUY})
        first = BacktestEngine().run(strategy, bars, symbol="XAUUSD", timeframe="M1")
        second = BacktestEngine().run(strategy, bars, symbol="XAUUSD", timeframe="M1")
        self.assertEqual(first.reproducibility_metadata, second.reproducibility_metadata)
        self.assertEqual(first.metrics.to_dict(), second.metrics.to_dict())

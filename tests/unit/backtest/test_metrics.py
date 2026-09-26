from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.backtest.metrics import calculate_metrics
from app.backtest.models import TradeOutcome, TradeRecord


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_trade(pnl: float, outcome: TradeOutcome, rr: float, offset: int) -> TradeRecord:
    ts = BASE + timedelta(minutes=offset)
    return TradeRecord(
        strategy_name="TEST",
        strategy_variant="TEST_V1",
        symbol="XAUUSD",
        timeframe="M15",
        direction="BUY",
        signal_timestamp=ts,
        entry_timestamp=ts,
        entry_price=100.0,
        stop_loss=95.0,
        target=100.0 + 5.0 * rr,
        planned_risk_reward=rr,
        quantity=1.0,
        exit_timestamp=ts + timedelta(minutes=1),
        exit_price=100.0 + pnl,
        outcome=outcome,
        r_multiple=pnl / 5.0,
        pnl_gross=pnl,
        commission=0.0,
        pnl_net=pnl,
        bars_held=1,
        exit_reason=outcome.value,
    )


class MetricsTests(unittest.TestCase):
    def test_metrics_are_mathematically_consistent(self) -> None:
        trades = (
            make_trade(20.0, TradeOutcome.WIN, 2.0, 0),
            make_trade(-10.0, TradeOutcome.LOSS, 1.0, 2),
        )
        metrics = calculate_metrics(trades, starting_capital=1_000.0)
        self.assertEqual(metrics.total_trades, 2)
        self.assertEqual(metrics.wins, 1)
        self.assertEqual(metrics.losses, 1)
        self.assertAlmostEqual(metrics.win_rate, 50.0)
        self.assertAlmostEqual(metrics.profit_factor or 0.0, 2.0)
        self.assertAlmostEqual(metrics.expectancy, 5.0)
        self.assertAlmostEqual(metrics.average_rr or 0.0, 1.5)
        self.assertAlmostEqual(metrics.max_drawdown, 10.0)
        self.assertAlmostEqual(metrics.average_trade, 5.0)
        self.assertAlmostEqual(metrics.net_return or 0.0, 1.0)


    def test_max_drawdown_includes_starting_equity(self) -> None:
        trades = (make_trade(-10.0, TradeOutcome.LOSS, 1.0, 0),)
        metrics = calculate_metrics(trades, starting_capital=1_000.0)
        self.assertAlmostEqual(metrics.max_drawdown, 10.0)

    def test_ambiguous_skips_are_not_counted_as_executed_trades(self) -> None:
        trades = (make_trade(0.0, TradeOutcome.AMBIGUOUS_SKIPPED, 2.0, 0),)
        metrics = calculate_metrics(trades, starting_capital=1_000.0)
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.ambiguous_skipped, 1)
        self.assertEqual(metrics.win_rate, 0.0)

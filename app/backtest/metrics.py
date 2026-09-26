"""Performance and stability metrics for the backtest engine."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Sequence

from app.backtest.models import BacktestMetrics, TradeOutcome, TradeRecord


def calculate_metrics(
    trades: Sequence[TradeRecord],
    *,
    starting_capital: float | None,
    stability_periods: int = 4,
) -> BacktestMetrics:
    """Calculate metrics from executed trades plus explicit ambiguity records."""
    records = tuple(trades)
    skipped = sum(trade.outcome == TradeOutcome.AMBIGUOUS_SKIPPED for trade in records)
    executed = tuple(trade for trade in records if trade.outcome != TradeOutcome.AMBIGUOUS_SKIPPED)
    total = len(executed)
    wins = sum(trade.outcome == TradeOutcome.WIN for trade in executed)
    losses = sum(trade.outcome == TradeOutcome.LOSS for trade in executed)
    expired = sum(trade.outcome == TradeOutcome.EXPIRED for trade in executed)

    win_rate = (wins / total * 100.0) if total else 0.0
    positive_pnl = sum(trade.pnl_net for trade in executed if trade.pnl_net > 0)
    negative_pnl_abs = sum(-trade.pnl_net for trade in executed if trade.pnl_net < 0)
    profit_factor = (positive_pnl / negative_pnl_abs) if negative_pnl_abs > 0 else None
    expectancy = mean(trade.pnl_net for trade in executed) if executed else 0.0
    rr_values = [trade.planned_risk_reward for trade in executed]
    average_rr = mean(rr_values) if rr_values else None
    average_trade = expectancy

    current_equity = starting_capital if starting_capital is not None else 0.0
    equity_curve: list[float] = [current_equity]
    for trade in executed:
        current_equity += trade.pnl_net
        equity_curve.append(current_equity)
    max_drawdown = _max_drawdown(equity_curve)
    net_return = (
        ((current_equity - starting_capital) / starting_capital) * 100.0
        if starting_capital is not None
        else None
    )

    return_values = [trade.pnl_net for trade in executed]
    return_stddev = pstdev(return_values) if len(return_values) > 1 else 0.0
    period_win_rates, period_positive = _stability_periods(executed, stability_periods)
    win_rate_stddev_by_period = pstdev(period_win_rates) if len(period_win_rates) > 1 else 0.0
    positive_period_ratio = (
        sum(period_positive) / len(period_positive) if period_positive else 0.0
    )

    return BacktestMetrics(
        total_trades=total,
        wins=wins,
        losses=losses,
        expired=expired,
        ambiguous_skipped=skipped,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        average_rr=average_rr,
        max_drawdown=max_drawdown,
        average_trade=average_trade,
        net_return=net_return,
        tp_rate=(wins / total * 100.0) if total else 0.0,
        sl_rate=(losses / total * 100.0) if total else 0.0,
        expiry_rate=(expired / total * 100.0) if total else 0.0,
        return_stddev=return_stddev,
        win_rate_stddev_by_period=win_rate_stddev_by_period,
        positive_period_ratio=positive_period_ratio,
    )


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    maximum = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        maximum = max(maximum, peak - value)
    return maximum


def _stability_periods(
    trades: Sequence[TradeRecord],
    periods: int,
) -> tuple[list[float], list[bool]]:
    if not trades or periods <= 0:
        return [], []
    bucket_count = min(periods, len(trades))
    sizes = [len(trades) // bucket_count] * bucket_count
    for index in range(len(trades) % bucket_count):
        sizes[index] += 1

    win_rates: list[float] = []
    positive: list[bool] = []
    offset = 0
    for size in sizes:
        bucket = trades[offset : offset + size]
        offset += size
        wins = sum(trade.outcome == TradeOutcome.WIN for trade in bucket)
        win_rates.append(wins / len(bucket) * 100.0)
        positive.append(sum(trade.pnl_net for trade in bucket) > 0)
    return win_rates, positive


__all__ = ["calculate_metrics"]

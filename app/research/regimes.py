"""Research-only market regime and chronological period summaries."""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable

from app.backtest.models import TradeOutcome, TradeRecord
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot


def summarize_trade_regimes(
    trades: Iterable[TradeRecord],
    analysis: MarketAnalysisSeries,
    *,
    volatility_lookback: int = 100,
) -> dict[str, Any]:
    """Group trade outcomes by decision-time trend and trailing volatility regime.

    Regimes are descriptive labels for research diagnostics only. Volatility is
    compared with a trailing median built from observations available at the trade
    timestamp, so future observations are never required to label a trade.
    """
    snapshots = analysis.snapshots
    by_time = {snapshot.timestamp: index for index, snapshot in enumerate(snapshots)}
    groups: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        index = by_time.get(trade.entry_timestamp)
        if index is None:
            continue
        snapshot = snapshots[index]
        trend = str(snapshot.get("trend.ema_alignment", "UNKNOWN"))
        volatility = _volatility_regime(snapshots, index, volatility_lookback)
        key = f"{trend}|{volatility}"
        groups.setdefault(key, []).append(trade)

    output: dict[str, Any] = {}
    for key in sorted(groups):
        bucket = groups[key]
        wins = sum(trade.outcome == TradeOutcome.WIN for trade in bucket)
        losses = sum(trade.outcome == TradeOutcome.LOSS for trade in bucket)
        pnl = sum(trade.pnl_net for trade in bucket)
        positive = sum(trade.pnl_net for trade in bucket if trade.pnl_net > 0)
        negative = sum(-trade.pnl_net for trade in bucket if trade.pnl_net < 0)
        pf = positive / negative if negative > 0 else None
        avg_r = sum(trade.r_multiple for trade in bucket) / len(bucket)
        output[key] = {
            "trades": len(bucket),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(bucket) * 100.0,
            "profit_factor": pf,
            "expectancy_r": avg_r,
            "net_pnl": pnl,
        }
    return output


def _volatility_regime(
    snapshots: tuple[MarketAnalysisSnapshot, ...],
    index: int,
    lookback: int,
) -> str:
    values: list[float] = []
    start = max(0, index - max(1, lookback) + 1)
    for snapshot in snapshots[start : index + 1]:
        value = snapshot.get("volatility.atr_pct")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    if len(values) < 5:
        return "VOL_UNKNOWN"
    baseline = median(values[:-1]) if len(values) > 1 else values[-1]
    current = values[-1]
    if baseline <= 0:
        return "VOL_UNKNOWN"
    if current < baseline * 0.80:
        return "VOL_LOW"
    if current > baseline * 1.20:
        return "VOL_HIGH"
    return "VOL_NORMAL"


__all__ = ["summarize_trade_regimes"]

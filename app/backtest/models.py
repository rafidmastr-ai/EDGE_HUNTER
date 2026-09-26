"""Immutable models for trades, metrics and serializable backtest results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class TradeOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    EXPIRED = "EXPIRED"
    AMBIGUOUS_SKIPPED = "AMBIGUOUS_SKIPPED"


@dataclass(frozen=True)
class TradeRecord:
    """One completed or explicitly skipped trade event."""

    strategy_name: str
    strategy_variant: str
    symbol: str
    timeframe: str
    direction: str
    signal_timestamp: datetime
    entry_timestamp: datetime
    entry_price: float
    stop_loss: float
    target: float
    planned_risk_reward: float
    quantity: float
    exit_timestamp: datetime
    exit_price: float
    outcome: TradeOutcome
    r_multiple: float
    pnl_gross: float
    commission: float
    pnl_net: float
    bars_held: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signal_timestamp"] = self.signal_timestamp.isoformat()
        payload["entry_timestamp"] = self.entry_timestamp.isoformat()
        payload["exit_timestamp"] = self.exit_timestamp.isoformat()
        payload["outcome"] = self.outcome.value
        return payload


@dataclass(frozen=True)
class BacktestMetrics:
    """Performance and stability measurements with explicit undefined cases."""

    total_trades: int
    wins: int
    losses: int
    expired: int
    ambiguous_skipped: int
    win_rate: float
    profit_factor: float | None
    expectancy: float
    average_rr: float | None
    max_drawdown: float
    average_trade: float
    net_return: float | None
    tp_rate: float
    sl_rate: float
    expiry_rate: float
    return_stddev: float
    win_rate_stddev_by_period: float
    positive_period_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """Complete exportable backtest output."""

    engine_version: str
    strategy_name: str
    strategy_variant: str
    symbol: str
    timeframe: str
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    trades: tuple[TradeRecord, ...]
    metrics: BacktestMetrics
    reproducibility_metadata: Mapping[str, Any] = field(default_factory=dict)
    generated_at_utc: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_timestamp": self.start_timestamp.isoformat() if self.start_timestamp else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "trades": [trade.to_dict() for trade in self.trades],
            "metrics": self.metrics.to_dict(),
            "reproducibility_metadata": dict(self.reproducibility_metadata),
            "generated_at_utc": self.generated_at_utc.isoformat() if self.generated_at_utc else None,
        }

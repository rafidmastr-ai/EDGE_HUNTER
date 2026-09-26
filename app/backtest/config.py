"""Configuration contracts for the Phase 05 backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntrabarPolicy(str, Enum):
    """Deterministic policy when a single OHLC candle touches TP and SL."""

    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"
    SKIP_TRADE = "skip_trade"


@dataclass(frozen=True)
class BacktestConfig:
    """Execution, cost, expiry and position-sizing configuration.

    Prices use the same units as canonical OHLC. ``spread`` is the full quoted
    spread and is applied adversely as half on entry and half on exit.
    ``commission_per_unit`` is charged once at entry and once at exit.
    ``contract_size`` converts manual lots into price units.
    """

    starting_capital: float | None = 10_000.0
    risk_per_trade_pct: float | None = None
    lot_size: float | None = None
    default_position_units: float = 1.0
    contract_size: float = 1.0
    spread: float = 0.0
    commission_per_unit: float = 0.0
    max_bars_in_trade: int | None = None
    intrabar_policy: IntrabarPolicy = IntrabarPolicy.STOP_FIRST

    def __post_init__(self) -> None:
        for name, value in {
            "spread": self.spread,
            "commission_per_unit": self.commission_per_unit,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.starting_capital is not None and self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive when supplied")
        if self.risk_per_trade_pct is not None and not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade_pct must be in (0, 100]")
        if self.lot_size is not None and self.lot_size <= 0:
            raise ValueError("lot_size must be positive when supplied")
        if self.default_position_units <= 0:
            raise ValueError("default_position_units must be positive")
        if self.contract_size <= 0:
            raise ValueError("contract_size must be positive")
        if self.max_bars_in_trade is not None and self.max_bars_in_trade <= 0:
            raise ValueError("max_bars_in_trade must be positive when supplied")
        if self.risk_per_trade_pct is not None and self.starting_capital is None:
            raise ValueError("risk-based sizing requires starting_capital")
        if self.risk_per_trade_pct is not None and self.lot_size is not None:
            raise ValueError("provide either risk_per_trade_pct or lot_size, not both")

    def position_units(
        self, entry: float, stop_loss: float, capital: float | None = None
    ) -> float:
        """Return deterministic position size in price units."""
        if self.lot_size is not None:
            return self.lot_size * self.contract_size
        if self.risk_per_trade_pct is None:
            return self.default_position_units
        equity = capital if capital is not None else self.starting_capital
        assert equity is not None
        risk_amount = equity * (self.risk_per_trade_pct / 100.0)
        price_risk = abs(entry - stop_loss)
        if price_risk <= 0:
            raise ValueError("entry and stop_loss must differ for risk-based sizing")
        return risk_amount / price_risk

    def to_dict(self) -> dict[str, object]:
        return {
            "starting_capital": self.starting_capital,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "lot_size": self.lot_size,
            "default_position_units": self.default_position_units,
            "contract_size": self.contract_size,
            "spread": self.spread,
            "commission_per_unit": self.commission_per_unit,
            "max_bars_in_trade": self.max_bars_in_trade,
            "intrabar_policy": self.intrabar_policy.value,
        }

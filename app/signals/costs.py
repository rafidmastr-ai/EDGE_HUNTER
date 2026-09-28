"""Cost-aware gate for strategy setups (spread + slippage + commission vs stop distance).

Research finding (EDGE HUNTER research phases 2 and 3, see
docs/COST_GATE_AND_DATA_CLEANING.md): on intraday timeframes the strategies'
structure-based stops are often so tight that the round-trip trading cost alone
consumes 0.3-1.2 R per trade on Forex, which no setup quality can recover.

The gate therefore withholds a setup (turns it into NO_SIGNAL) when

    cost_r = estimated round-trip cost / |entry - stop_loss|  >  max_cost_r

which is the same as requiring a minimum stop distance of cost / max_cost_r.
It never moves or invents prices; accepted setups are annotated with the cost
estimate so the UI/API can show it. Cost figures are conservative estimates for
typical retail ECN/STP conditions, not a broker quote; override per deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from app.strategies.models import SignalDirection, SignalState, StrategySignal

FOREX_CODES = frozenset({"EUR", "GBP", "USD", "JPY", "CHF", "AUD", "CAD", "NZD", "SGD", "HKD", "NOK", "SEK", "DKK", "PLN", "TRY", "ZAR", "MXN", "CNH"})
CRYPTO_CODES = frozenset({"BTC", "ETH", "LTC", "XRP", "BCH", "SOL", "ADA", "DOGE", "DOT", "BNB", "AVAX", "LINK", "MATIC", "TRX", "XLM"})

# Typical spread in pips for liquid pairs; slippage and commission are added below.
FOREX_SPREAD_PIPS: Mapping[str, float] = {
    "EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 0.9, "AUDUSD": 1.0, "NZDUSD": 1.8, "USDCAD": 1.3,
    "USDCHF": 1.3, "EURGBP": 1.2, "EURJPY": 1.4, "GBPJPY": 2.2, "EURCHF": 1.6, "AUDJPY": 1.6,
}
DEFAULT_FOREX_SPREAD_PIPS = 2.5
FOREX_SLIPPAGE_PIPS = 0.3
FOREX_COMMISSION_PIPS = 0.7  # ~7 USD per standard lot round trip
# Round trip in price units (spread + slippage + commission).
METAL_COST: Mapping[str, float] = {"XAU": 0.42, "XAG": 0.045, "XPT": 2.5, "XPD": 5.0}
CRYPTO_COST_FRACTION = 0.001  # 0.10% of price round trip
OTHER_COST_FRACTION = 0.0005  # indices / stocks / unknown instruments


def split_symbol(symbol: str) -> tuple[str, str]:
    text = symbol.upper().replace(" ", "").replace("-", "/").replace("_", "/")
    if "/" in text:
        base, _, quote = text.partition("/")
        return base, quote
    return text[:3], text[3:]


@dataclass(frozen=True)
class CostEstimate:
    value: float
    basis: str


@dataclass(frozen=True)
class CostModel:
    """Deterministic round-trip cost estimate per symbol in price units."""

    forex_spread_pips: Mapping[str, float] = field(default_factory=lambda: dict(FOREX_SPREAD_PIPS))
    default_forex_spread_pips: float = DEFAULT_FOREX_SPREAD_PIPS
    forex_slippage_pips: float = FOREX_SLIPPAGE_PIPS
    forex_commission_pips: float = FOREX_COMMISSION_PIPS
    metal_cost: Mapping[str, float] = field(default_factory=lambda: dict(METAL_COST))
    crypto_cost_fraction: float = CRYPTO_COST_FRACTION
    other_cost_fraction: float = OTHER_COST_FRACTION

    def estimate(self, symbol: str, price: float) -> CostEstimate:
        base, quote = split_symbol(symbol)
        if base in self.metal_cost:
            return CostEstimate(self.metal_cost[base], "metal_fixed_estimate")
        if base in FOREX_CODES and quote in FOREX_CODES:
            pip = 0.01 if quote == "JPY" else 0.0001
            spread = self.forex_spread_pips.get(base + quote, self.default_forex_spread_pips)
            pips = spread + self.forex_slippage_pips + self.forex_commission_pips
            return CostEstimate(pips * pip, f"forex_{pips:.1f}_pips_round_trip")
        if base in CRYPTO_CODES:
            return CostEstimate(abs(price) * self.crypto_cost_fraction, "crypto_fraction_of_price")
        return CostEstimate(abs(price) * self.other_cost_fraction, "generic_fraction_of_price")


@dataclass(frozen=True)
class CostGate:
    """Withhold setups whose trading cost is too large relative to their risk."""

    max_cost_r: float = 0.10
    model: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_cost_r <= 1.0:
            raise ValueError("max_cost_r must be in (0, 1]")

    def evaluate(self, signal: StrategySignal) -> dict:
        """Cost metadata for an actionable setup (no decision)."""
        assert signal.entry is not None and signal.stop_loss is not None and signal.risk_reward is not None
        risk = abs(float(signal.entry) - float(signal.stop_loss))
        estimate = self.model.estimate(signal.symbol, float(signal.entry))
        cost_r = estimate.value / risk if risk > 0 else float("inf")
        return {
            "cost_estimate": round(estimate.value, 8),
            "cost_basis": estimate.basis,
            "cost_r": round(cost_r, 4) if cost_r != float("inf") else None,
            "cost_max_r": self.max_cost_r,
            "min_stop_distance": round(estimate.value / self.max_cost_r, 8),
            "net_risk_reward_after_cost": round(float(signal.risk_reward) - cost_r, 4) if cost_r != float("inf") else None,
        }

    def apply(self, signals: Sequence[StrategySignal]) -> tuple[StrategySignal, ...]:
        output: list[StrategySignal] = []
        for signal in signals:
            if signal.state != SignalState.SIGNAL:
                output.append(signal)
                continue
            meta = self.evaluate(signal)
            cost_r = meta["cost_r"]
            if cost_r is not None and cost_r <= self.max_cost_r:
                output.append(replace(signal, metadata={**dict(signal.metadata), **meta, "cost_decision": "accepted"}))
                continue
            shown = "inf" if cost_r is None else f"{cost_r:.2f}"
            output.append(
                StrategySignal(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    direction=SignalDirection.NO_SIGNAL,
                    state=SignalState.NO_SIGNAL,
                    entry=None,
                    stop_loss=None,
                    target=None,
                    risk_reward=None,
                    entry_logic="Setup withheld: estimated trading cost is too large relative to the stop distance.",
                    invalidation="No active setup.",
                    stop_loss_logic="No stop-loss generated.",
                    target_logic="No target generated.",
                    evidence=(f"cost_gate_rejected:cost_r={shown}>{self.max_cost_r:.2f}",),
                    score_inputs={},
                    strategy_name=signal.strategy_name,
                    variant=signal.variant,
                    metadata={
                        **{key: value for key, value in dict(signal.metadata).items() if str(key).startswith("ml_")},
                        **meta,
                        "cost_decision": "rejected",
                        "original_direction": signal.direction.value,
                    },
                )
            )
        return tuple(output)


__all__ = ["CostEstimate", "CostGate", "CostModel", "split_symbol"]

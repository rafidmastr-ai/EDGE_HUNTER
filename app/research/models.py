"""Research contracts for systematic strategy comparison in Phase 06."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

from app.backtest.models import BacktestResult
from app.strategies.models import StrategyConfig


@dataclass(frozen=True)
class ResearchVariant:
    """Named parameter configuration used as one reproducible research experiment."""

    name: str
    description: str
    strategy_config: StrategyConfig
    family: str = "initial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "strategy_config": asdict(self.strategy_config),
        }


@dataclass(frozen=True)
class ResearchMatrix:
    """Cartesian experiment matrix. No ordering implies a performance ranking."""

    symbols: tuple[str, ...] = ("XAUUSD", "GBPUSD")
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4")
    strategy_names: tuple[str, ...] = ("Classic", "SMC", "ICT")
    variants: tuple[ResearchVariant, ...] = ()
    is_fraction: float = 0.70
    stability_periods: int = 4

    def __post_init__(self) -> None:
        if not 0.5 <= self.is_fraction < 1.0:
            raise ValueError("is_fraction must be in [0.5, 1.0)")
        if self.stability_periods < 2:
            raise ValueError("stability_periods must be >= 2")
        if not self.symbols or not self.timeframes or not self.strategy_names:
            raise ValueError("symbols, timeframes and strategy_names must not be empty")
        if not self.variants:
            raise ValueError("research matrix requires at least one variant")

    @classmethod
    def default(cls) -> "ResearchMatrix":
        """Return the initial non-optimized matrix.

        The variants deliberately cover the practical 1.5–2.0 R:R range and a
        small number of existing Phase 04 parameter variations. They are research
        configurations, not assertions of optimality.
        """
        return cls(variants=default_variants())

    @classmethod
    def quick(cls) -> "ResearchMatrix":
        """Small deterministic matrix for local smoke testing."""
        variants = tuple(default_variants()[:3])
        return cls(
            symbols=("XAUUSD", "GBPUSD"),
            timeframes=("M1", "M15", "H1"),
            strategy_names=("Classic", "SMC", "ICT"),
            variants=variants,
        )

    def experiment_count(self) -> int:
        return len(self.symbols) * len(self.timeframes) * len(self.strategy_names) * len(self.variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "strategy_names": list(self.strategy_names),
            "variants": [variant.to_dict() for variant in self.variants],
            "is_fraction": self.is_fraction,
            "stability_periods": self.stability_periods,
        }


@dataclass(frozen=True)
class ResearchScreeningConfig:
    """Configurable evidence gates used to nominate Phase 07 candidates.

    These are screening defaults for research reproducibility, not universal
    trading rules and not optimization targets.
    """

    minimum_is_trades: int = 30
    minimum_oos_trades: int = 30
    minimum_oos_profit_factor: float = 1.0
    minimum_oos_expectancy_r: float = 0.0
    minimum_oos_positive_period_ratio: float = 0.50
    maximum_oos_win_rate_stddev: float = 25.0
    maximum_signal_to_trade_ratio: float = 5.0
    maximum_signals_per_1000_bars: float = 50.0

    def __post_init__(self) -> None:
        if self.minimum_is_trades < 0 or self.minimum_oos_trades < 0:
            raise ValueError("minimum trade counts must be >= 0")
        if self.minimum_oos_profit_factor < 0:
            raise ValueError("minimum_oos_profit_factor must be >= 0")
        if self.minimum_oos_positive_period_ratio < 0 or self.minimum_oos_positive_period_ratio > 1:
            raise ValueError("minimum_oos_positive_period_ratio must be in [0, 1]")
        if self.maximum_oos_win_rate_stddev < 0:
            raise ValueError("maximum_oos_win_rate_stddev must be >= 0")
        if self.maximum_signal_to_trade_ratio < 0 or self.maximum_signals_per_1000_bars < 0:
            raise ValueError("overtrading thresholds must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    """Comparable metrics independent of symbol price scale."""

    total_trades: int
    wins: int
    losses: int
    expired: int
    ambiguous_skipped: int
    win_rate: float
    profit_factor: float | None
    expectancy: float
    expectancy_r: float
    average_rr: float | None
    max_drawdown: float
    max_drawdown_pct: float | None
    average_trade: float
    net_return: float | None
    tp_rate: float
    sl_rate: float
    expiry_rate: float
    return_stddev: float
    win_rate_stddev_by_period: float
    positive_period_ratio: float

    @classmethod
    def from_result(cls, result: BacktestResult) -> "MetricSummary":
        trades = tuple(
            trade for trade in result.trades
            if trade.outcome.value != "AMBIGUOUS_SKIPPED"
        )
        average_r = (
            sum(trade.r_multiple for trade in trades) / len(trades)
            if trades else 0.0
        )
        starting_capital = result.reproducibility_metadata.get("execution_config", {}).get("starting_capital")
        dd_pct = None
        if isinstance(starting_capital, (int, float)) and starting_capital:
            dd_pct = result.metrics.max_drawdown / float(starting_capital) * 100.0
        metrics = result.metrics
        return cls(
            total_trades=metrics.total_trades,
            wins=metrics.wins,
            losses=metrics.losses,
            expired=metrics.expired,
            ambiguous_skipped=metrics.ambiguous_skipped,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            expectancy=metrics.expectancy,
            expectancy_r=average_r,
            average_rr=metrics.average_rr,
            max_drawdown=metrics.max_drawdown,
            max_drawdown_pct=dd_pct,
            average_trade=metrics.average_trade,
            net_return=metrics.net_return,
            tp_rate=metrics.tp_rate,
            sl_rate=metrics.sl_rate,
            expiry_rate=metrics.expiry_rate,
            return_stddev=metrics.return_stddev,
            win_rate_stddev_by_period=metrics.win_rate_stddev_by_period,
            positive_period_ratio=metrics.positive_period_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalStats:
    """Decision-state counts captured while the backtest runs."""

    evaluations: int
    signals_generated: int
    no_signals: int
    conflicts: int
    insufficient_data: int
    signals_per_1000_bars: float
    signal_to_trade_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentAssessment:
    """Evidence tags and Phase 07 eligibility without performance ranking."""

    flags: tuple[str, ...]
    candidate_for_phase07: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchExperiment:
    """One reproducible IS/OOS strategy-comparison experiment."""

    experiment_id: str
    symbol: str
    timeframe: str
    strategy_name: str
    strategy_variant: str
    variant_name: str
    variant_config: Mapping[str, Any]
    is_metrics: MetricSummary
    oos_metrics: MetricSummary
    is_signal_stats: SignalStats
    oos_signal_stats: SignalStats
    assessment: ExperimentAssessment
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "variant_name": self.variant_name,
            "variant_config": dict(self.variant_config),
            "is_metrics": self.is_metrics.to_dict(),
            "oos_metrics": self.oos_metrics.to_dict(),
            "is_signal_stats": self.is_signal_stats.to_dict(),
            "oos_signal_stats": self.oos_signal_stats.to_dict(),
            "assessment": self.assessment.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResearchRun:
    """Aggregated research output ready for machine and human reports."""

    research_engine_version: str
    run_id: str
    generated_at_utc: datetime
    matrix: ResearchMatrix
    screening: ResearchScreeningConfig
    experiments: tuple[ResearchExperiment, ...]
    dataset_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> tuple[ResearchExperiment, ...]:
        return tuple(exp for exp in self.experiments if exp.assessment.candidate_for_phase07)

    def to_dict(self) -> dict[str, Any]:
        return {
            "research_engine_version": self.research_engine_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "matrix": self.matrix.to_dict(),
            "screening": self.screening.to_dict(),
            "experiments": [exp.to_dict() for exp in self.experiments],
            "dataset_metadata": dict(self.dataset_metadata),
            "candidate_count": len(self.candidates),
        }


def default_variants() -> tuple[ResearchVariant, ...]:
    """Initial Phase 06 research variants; none is labeled optimal."""
    return (
        ResearchVariant(
            "BASE_DYNAMIC_1P5_2P0",
            "Baseline Phase 04 parameters with dynamic R:R bounded to 1.5–2.0.",
            StrategyConfig(min_rr=1.5, max_rr=2.0, swing_lookback=5, minimum_body_ratio=0.55),
        ),
        ResearchVariant(
            "RR_FIXED_1P50",
            "Fixed R:R at 1.50 using the existing adaptive policy with equal bounds.",
            StrategyConfig(min_rr=1.5, max_rr=1.5, swing_lookback=5, minimum_body_ratio=0.55),
        ),
        ResearchVariant(
            "RR_FIXED_1P75",
            "Fixed R:R at 1.75 using the existing adaptive policy with equal bounds.",
            StrategyConfig(min_rr=1.75, max_rr=1.75, swing_lookback=5, minimum_body_ratio=0.55),
        ),
        ResearchVariant(
            "RR_FIXED_2P00",
            "Fixed R:R at 2.00 using the existing adaptive policy with equal bounds.",
            StrategyConfig(min_rr=2.0, max_rr=2.0, swing_lookback=5, minimum_body_ratio=0.55),
        ),
        ResearchVariant(
            "BODY_0P65_DYNAMIC_1P5_2P0",
            "Stricter existing candle-body confirmation with the same dynamic R:R range.",
            StrategyConfig(min_rr=1.5, max_rr=2.0, swing_lookback=5, minimum_body_ratio=0.65),
        ),
        ResearchVariant(
            "SWING_8_DYNAMIC_1P5_2P0",
            "Longer existing structure lookback with the same dynamic R:R range.",
            StrategyConfig(min_rr=1.5, max_rr=2.0, swing_lookback=8, minimum_body_ratio=0.55),
        ),
    )


__all__ = [
    "ExperimentAssessment",
    "MetricSummary",
    "ResearchExperiment",
    "ResearchMatrix",
    "ResearchRun",
    "ResearchScreeningConfig",
    "ResearchVariant",
    "SignalStats",
    "default_variants",
]

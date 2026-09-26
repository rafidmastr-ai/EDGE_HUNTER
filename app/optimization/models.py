"""Immutable contracts for deterministic optimization and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable parameter with a documented research reason."""

    name: str
    values: tuple[Any, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")
        if not self.values:
            raise ValueError(f"parameter {self.name!r} requires at least one value")
        if not self.reason.strip():
            raise ValueError(f"parameter {self.name!r} requires a documented research reason")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "values": list(self.values), "reason": self.reason}


@dataclass(frozen=True)
class OptimizationSpace:
    """Search space built only from parameters justified by Phase 06 research."""

    parameters: tuple[ParameterSpec, ...]
    method: str = "grid"
    max_trials: int = 200
    seed: int = 20260924

    def __post_init__(self) -> None:
        if self.method not in {"grid", "random"}:
            raise ValueError("method must be 'grid' or 'random'")
        if not self.parameters:
            raise ValueError("optimization space requires at least one parameter")
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "max_trials": self.max_trials,
            "seed": self.seed,
            "parameters": [item.to_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class SplitPlan:
    """Chronological train/validation/OOS split using full-history indices."""

    train_end: int
    validation_end: int
    total_bars: int

    def __post_init__(self) -> None:
        if self.total_bars < 3:
            raise ValueError("at least three bars are required for train/validation/OOS")
        if not 1 <= self.train_end < self.validation_end < self.total_bars:
            raise ValueError("split indices must satisfy 0 < train_end < validation_end < total_bars")

    @property
    def train_range(self) -> tuple[int, int]:
        return 0, self.train_end

    @property
    def validation_range(self) -> tuple[int, int]:
        return self.train_end, self.validation_end

    @property
    def oos_range(self) -> tuple[int, int]:
        return self.validation_end, self.total_bars

    def to_dict(self) -> dict[str, int]:
        return {
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "total_bars": self.total_bars,
        }


@dataclass(frozen=True)
class OptimizationConfig:
    """Search and robustness gates; values are research controls, not guarantees."""

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    min_validation_trades: int = 10
    min_oos_trades: int = 10
    min_validation_expectancy_r: float = 0.0
    min_oos_expectancy_r: float = 0.0
    min_validation_profit_factor: float = 1.0
    min_oos_profit_factor: float = 1.0
    min_positive_validation_period_ratio: float = 0.50
    min_positive_oos_period_ratio: float = 0.50
    max_validation_drawdown_pct: float | None = None
    max_oos_drawdown_pct: float | None = None
    max_train_to_validation_expectancy_drop: float = 1.0
    min_oos_retention_ratio: float = 0.50
    sensitivity_neighbor_count: int = 1
    max_local_performance_deterioration: float = 0.50
    walk_forward_folds: int = 3
    candidate_pool_size: int = 5
    top_train_pool_size: int = 8

    def __post_init__(self) -> None:
        if not 0.50 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        for name in ("min_validation_trades", "min_oos_trades"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in (
            "sensitivity_neighbor_count",
            "walk_forward_folds",
            "candidate_pool_size",
            "top_train_pool_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in (
            "min_positive_validation_period_ratio",
            "min_positive_oos_period_ratio",
            "max_train_to_validation_expectancy_drop",
            "min_oos_retention_ratio",
            "max_local_performance_deterioration",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.min_positive_validation_period_ratio > 1 or self.min_positive_oos_period_ratio > 1:
            raise ValueError("positive period ratios must be <= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationCandidate:
    """One strategy target admitted from Phase 06 research."""

    strategy_name: str
    base_config: Mapping[str, Any]
    reference_experiment_id: str | None = None
    research_reason: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if not self.base_config:
            raise ValueError("base_config must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "base_config": dict(self.base_config),
            "reference_experiment_id": self.reference_experiment_id,
            "research_reason": self.research_reason,
        }


@dataclass(frozen=True)
class PeriodMetrics:
    """Compact comparable metrics for one period."""

    total_trades: int
    expectancy_r: float
    profit_factor: float | None
    max_drawdown_pct: float | None
    positive_period_ratio: float
    win_rate: float
    average_rr: float | None
    net_return: float | None

    @classmethod
    def from_metric_summary(cls, metrics: Any) -> "PeriodMetrics":
        return cls(
            total_trades=int(metrics.total_trades),
            expectancy_r=float(metrics.expectancy_r),
            profit_factor=None if metrics.profit_factor is None else float(metrics.profit_factor),
            max_drawdown_pct=None if metrics.max_drawdown_pct is None else float(metrics.max_drawdown_pct),
            positive_period_ratio=float(metrics.positive_period_ratio),
            win_rate=float(metrics.win_rate),
            average_rr=None if metrics.average_rr is None else float(metrics.average_rr),
            net_return=None if metrics.net_return is None else float(metrics.net_return),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessSummary:
    """Sensitivity and cross-case evidence for one evaluated configuration."""

    sensitivity_score: float | None
    neighbor_count: int
    worst_neighbor_deterioration: float | None
    parameter_stability: float | None
    positive_case_ratio: float | None
    walk_forward_positive_fold_ratio: float | None
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationEvaluation:
    """All measurements for one configuration across train/validation/OOS."""

    evaluation_id: str
    strategy_name: str
    config: Mapping[str, Any]
    train: PeriodMetrics
    validation: PeriodMetrics
    oos: PeriodMetrics
    train_score: float
    validation_score: float
    oos_score: float
    validation_eligible: bool
    final_candidate: bool
    rejection_reasons: tuple[str, ...]
    robustness: RobustnessSummary
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "strategy_name": self.strategy_name,
            "config": dict(self.config),
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "oos": self.oos.to_dict(),
            "train_score": self.train_score,
            "validation_score": self.validation_score,
            "oos_score": self.oos_score,
            "validation_eligible": self.validation_eligible,
            "final_candidate": self.final_candidate,
            "rejection_reasons": list(self.rejection_reasons),
            "robustness": self.robustness.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OptimizationResult:
    """Serializable output of a complete Phase 07 optimization run."""

    engine_version: str
    run_id: str
    generated_at_utc: datetime
    strategy_name: str
    candidate_reference: Mapping[str, Any]
    space: OptimizationSpace
    config: OptimizationConfig
    split: SplitPlan
    evaluations: tuple[OptimizationEvaluation, ...]
    final_candidates: tuple[OptimizationEvaluation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "strategy_name": self.strategy_name,
            "candidate_reference": dict(self.candidate_reference),
            "space": self.space.to_dict(),
            "config": self.config.to_dict(),
            "split": self.split.to_dict(),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "final_candidates": [item.to_dict() for item in self.final_candidates],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardSummary:
    folds: tuple[WalkForwardFold, ...]
    fold_results: tuple[Mapping[str, Any], ...]
    positive_fold_ratio: float
    mean_expectancy_r: float
    mean_profit_factor: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "fold_results": [dict(item) for item in self.fold_results],
            "positive_fold_ratio": self.positive_fold_ratio,
            "mean_expectancy_r": self.mean_expectancy_r,
            "mean_profit_factor": self.mean_profit_factor,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "OptimizationCandidate",
    "OptimizationConfig",
    "OptimizationEvaluation",
    "OptimizationResult",
    "OptimizationSpace",
    "ParameterSpec",
    "PeriodMetrics",
    "RobustnessSummary",
    "SplitPlan",
    "WalkForwardFold",
    "WalkForwardSummary",
    "utc_now",
]

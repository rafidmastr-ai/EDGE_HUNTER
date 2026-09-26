"""Multi-objective scoring for optimization search and validation gates."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from app.optimization.models import OptimizationConfig, PeriodMetrics


def _bounded(value: float, low: float, high: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(low, min(high, value))


def _pf_component(pf: float | None) -> float:
    if pf is None:
        return 0.0
    return _bounded((pf - 1.0) / 1.0, -1.0, 1.0)


def _dd_component(dd_pct: float | None) -> float:
    if dd_pct is None:
        return 0.0
    # 0% DD -> 1.0, 50%+ -> -1.0.
    return _bounded(1.0 - (abs(dd_pct) / 25.0), -1.0, 1.0)


def _trades_component(total_trades: int) -> float:
    # Enough activity without rewarding unlimited frequency.
    return _bounded((float(total_trades) - 5.0) / 45.0, -1.0, 1.0)


def _stability_component(positive_period_ratio: float) -> float:
    return _bounded(positive_period_ratio * 2.0 - 1.0, -1.0, 1.0)


def score_period(metrics: PeriodMetrics) -> float:
    """Composite search score using several dimensions, never net return alone."""
    expectancy = _bounded(metrics.expectancy_r, -1.0, 1.0)
    pf = _pf_component(metrics.profit_factor)
    dd = _dd_component(metrics.max_drawdown_pct)
    trades = _trades_component(metrics.total_trades)
    stability = _stability_component(metrics.positive_period_ratio)
    rr = _bounded(((metrics.average_rr or 0.0) - 1.5) / 0.5, -1.0, 1.0)
    win = _bounded((metrics.win_rate - 50.0) / 50.0, -1.0, 1.0)
    return (
        0.28 * expectancy
        + 0.20 * pf
        + 0.16 * dd
        + 0.14 * stability
        + 0.08 * trades
        + 0.08 * rr
        + 0.06 * win
    )


def score_evidence(metrics: PeriodMetrics) -> float:
    """Validation-friendly score excluding sample-size inflation."""
    expectancy = _bounded(metrics.expectancy_r, -1.0, 1.0)
    pf = _pf_component(metrics.profit_factor)
    dd = _dd_component(metrics.max_drawdown_pct)
    stability = _stability_component(metrics.positive_period_ratio)
    rr = _bounded(((metrics.average_rr or 0.0) - 1.5) / 0.5, -1.0, 1.0)
    return 0.35 * expectancy + 0.25 * pf + 0.20 * dd + 0.15 * stability + 0.05 * rr


def retention_ratio(train_score: float, later_score: float) -> float:
    """Return a bounded deterioration/retention measure for positive train scores."""
    if train_score <= 0:
        return 0.0 if later_score <= 0 else 1.0
    return _bounded(later_score / train_score, -1.0, 2.0)


def is_gate_eligible(
    train: PeriodMetrics,
    validation: PeriodMetrics,
    oos: PeriodMetrics,
    config: OptimizationConfig,
    *,
    robustness_flags: tuple[str, ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if validation.total_trades < config.min_validation_trades:
        reasons.append("validation_sample_too_small")
    if oos.total_trades < config.min_oos_trades:
        reasons.append("oos_sample_too_small")
    if validation.expectancy_r < config.min_validation_expectancy_r:
        reasons.append("validation_expectancy_below_threshold")
    if oos.expectancy_r < config.min_oos_expectancy_r:
        reasons.append("oos_expectancy_below_threshold")
    if validation.profit_factor is None or validation.profit_factor < config.min_validation_profit_factor:
        reasons.append("validation_profit_factor_below_threshold")
    if oos.profit_factor is None or oos.profit_factor < config.min_oos_profit_factor:
        reasons.append("oos_profit_factor_below_threshold")
    if validation.positive_period_ratio < config.min_positive_validation_period_ratio:
        reasons.append("validation_stability_below_threshold")
    if oos.positive_period_ratio < config.min_positive_oos_period_ratio:
        reasons.append("oos_stability_below_threshold")
    if (
        config.max_validation_drawdown_pct is not None
        and validation.max_drawdown_pct is not None
        and validation.max_drawdown_pct > config.max_validation_drawdown_pct
    ):
        reasons.append("validation_drawdown_above_threshold")
    if (
        config.max_oos_drawdown_pct is not None
        and oos.max_drawdown_pct is not None
        and oos.max_drawdown_pct > config.max_oos_drawdown_pct
    ):
        reasons.append("oos_drawdown_above_threshold")
    train_to_val_drop = train.expectancy_r - validation.expectancy_r
    if train.expectancy_r > 0 and train_to_val_drop / train.expectancy_r > config.max_train_to_validation_expectancy_drop:
        reasons.append("train_validation_expectancy_deterioration")
    later_retention = retention_ratio(score_period(train), score_period(oos))
    if later_retention < config.min_oos_retention_ratio:
        reasons.append("oos_score_retention_below_threshold")
    reasons.extend(robustness_flags)
    return not reasons, tuple(dict.fromkeys(reasons))


__all__ = ["is_gate_eligible", "retention_ratio", "score_evidence", "score_period"]

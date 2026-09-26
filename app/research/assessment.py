"""Multi-metric evidence screening for Phase 06 research outputs."""

from __future__ import annotations

from app.research.models import ExperimentAssessment, MetricSummary, ResearchScreeningConfig, SignalStats


def assess_experiment(
    is_metrics: MetricSummary,
    oos_metrics: MetricSummary,
    is_signal_stats: SignalStats,
    oos_signal_stats: SignalStats,
    config: ResearchScreeningConfig,
) -> ExperimentAssessment:
    flags: list[str] = []
    reasons: list[str] = []

    if is_metrics.total_trades < config.minimum_is_trades or oos_metrics.total_trades < config.minimum_oos_trades:
        flags.append("INSUFFICIENT_SAMPLE")
        reasons.append(
            f"trade sample below screening minimum (IS={is_metrics.total_trades}, OOS={oos_metrics.total_trades})"
        )

    overtraded = (
        oos_signal_stats.signal_to_trade_ratio is not None
        and oos_signal_stats.signal_to_trade_ratio > config.maximum_signal_to_trade_ratio
    ) or oos_signal_stats.signals_per_1000_bars > config.maximum_signals_per_1000_bars
    if overtraded:
        flags.append("OVERTRADED")
        reasons.append("signal frequency exceeded configured research screening threshold")

    unstable = (
        oos_metrics.win_rate_stddev_by_period > config.maximum_oos_win_rate_stddev
        or oos_metrics.positive_period_ratio < config.minimum_oos_positive_period_ratio
    )
    if unstable:
        flags.append("UNSTABLE")
        reasons.append("out-of-sample period consistency did not meet the stability screen")

    data_sensitive = (
        is_metrics.total_trades >= config.minimum_is_trades
        and oos_metrics.total_trades >= config.minimum_oos_trades
        and is_metrics.expectancy_r > 0
        and oos_metrics.expectancy_r <= 0
    )
    if data_sensitive:
        flags.append("DATA_SENSITIVE")
        reasons.append("positive in-sample expectancy did not persist out of sample")

    pf_ok = oos_metrics.profit_factor is not None and oos_metrics.profit_factor >= config.minimum_oos_profit_factor
    expectancy_ok = oos_metrics.expectancy_r >= config.minimum_oos_expectancy_r
    stability_ok = oos_metrics.positive_period_ratio >= config.minimum_oos_positive_period_ratio
    sample_ok = (
        is_metrics.total_trades >= config.minimum_is_trades
        and oos_metrics.total_trades >= config.minimum_oos_trades
    )
    not_negative_flagged = not any(flag in flags for flag in ("OVERTRADED", "UNSTABLE", "DATA_SENSITIVE", "INSUFFICIENT_SAMPLE"))

    candidate = sample_ok and pf_ok and expectancy_ok and stability_ok and not_negative_flagged
    if candidate:
        flags.append("PROMISING")
        reasons.append("passed the configured multi-metric evidence screen and is eligible for Phase 07 review")
    else:
        if not flags:
            flags.append("NOT_CANDIDATE")
        if not pf_ok:
            reasons.append("OOS profit factor did not meet the screening threshold")
        if not expectancy_ok:
            reasons.append("OOS expectancy in R did not meet the screening threshold")
        if not stability_ok:
            reasons.append("OOS positive-period ratio did not meet the screening threshold")

    return ExperimentAssessment(
        flags=tuple(dict.fromkeys(flags)),
        candidate_for_phase07=candidate,
        reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = ["assess_experiment"]

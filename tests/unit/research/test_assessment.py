from __future__ import annotations

import unittest

from app.research.assessment import assess_experiment
from app.research.models import MetricSummary, ResearchScreeningConfig, SignalStats


def metrics(trades: int, exp_r: float, pf: float | None, positive_ratio: float = 0.75, wr_std: float = 10.0) -> MetricSummary:
    return MetricSummary(
        total_trades=trades, wins=trades // 2, losses=trades - trades // 2, expired=0,
        ambiguous_skipped=0, win_rate=50.0, profit_factor=pf, expectancy=exp_r,
        expectancy_r=exp_r, average_rr=1.75, max_drawdown=100.0, max_drawdown_pct=1.0,
        average_trade=exp_r, net_return=5.0, tp_rate=50.0, sl_rate=50.0, expiry_rate=0.0,
        return_stddev=1.0, win_rate_stddev_by_period=wr_std, positive_period_ratio=positive_ratio,
    )


def signal_stats(signals: int, trades: int, bars: int = 1000) -> SignalStats:
    return SignalStats(
        evaluations=bars, signals_generated=signals, no_signals=bars-signals,
        conflicts=0, insufficient_data=0, signals_per_1000_bars=signals / bars * 1000,
        signal_to_trade_ratio=signals / trades if trades else None,
    )


class AssessmentTests(unittest.TestCase):
    def test_candidate_requires_multiple_evidence_gates(self) -> None:
        assessment = assess_experiment(
            metrics(50, 0.10, 1.2), metrics(50, 0.08, 1.1),
            signal_stats(50, 50), signal_stats(40, 50), ResearchScreeningConfig(),
        )
        self.assertTrue(assessment.candidate_for_phase07)
        self.assertIn("PROMISING", assessment.flags)

    def test_negative_oos_after_positive_is_is_data_sensitive(self) -> None:
        assessment = assess_experiment(
            metrics(50, 0.10, 1.2), metrics(50, -0.05, 0.9),
            signal_stats(50, 50), signal_stats(40, 50), ResearchScreeningConfig(),
        )
        self.assertFalse(assessment.candidate_for_phase07)
        self.assertIn("DATA_SENSITIVE", assessment.flags)

    def test_high_signal_frequency_is_flagged_overtraded(self) -> None:
        assessment = assess_experiment(
            metrics(50, 0.10, 1.2), metrics(50, 0.08, 1.1),
            signal_stats(50, 50), signal_stats(600, 50), ResearchScreeningConfig(),
        )
        self.assertIn("OVERTRADED", assessment.flags)
        self.assertFalse(assessment.candidate_for_phase07)

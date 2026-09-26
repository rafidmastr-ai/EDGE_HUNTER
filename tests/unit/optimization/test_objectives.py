from __future__ import annotations

import unittest

from app.optimization.models import OptimizationConfig, PeriodMetrics
from app.optimization.objectives import is_gate_eligible, retention_ratio, score_period


def metrics(
    trades=50, exp=0.2, pf=1.4, dd=5.0, stability=0.75, wr=55.0, rr=1.75, ret=4.0
):
    return PeriodMetrics(trades, exp, pf, dd, stability, wr, rr, ret)


class OptimizationObjectiveTests(unittest.TestCase):
    def test_score_uses_multiple_dimensions(self) -> None:
        strong = score_period(metrics())
        weaker = score_period(metrics(exp=0.0, pf=1.0, stability=0.5))
        self.assertGreater(strong, weaker)

    def test_retention_is_deterministic(self) -> None:
        self.assertAlmostEqual(retention_ratio(1.0, 0.75), 0.75)

    def test_gate_rejects_small_oos_sample(self) -> None:
        cfg = OptimizationConfig(min_validation_trades=10, min_oos_trades=20)
        ok, reasons = is_gate_eligible(metrics(), metrics(trades=12), metrics(trades=5), cfg)
        self.assertFalse(ok)
        self.assertIn("oos_sample_too_small", reasons)

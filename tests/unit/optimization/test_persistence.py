from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.optimization.models import (
    OptimizationCandidate,
    OptimizationConfig,
    OptimizationEvaluation,
    OptimizationResult,
    OptimizationSpace,
    ParameterSpec,
    PeriodMetrics,
    RobustnessSummary,
    SplitPlan,
)
from app.optimization.persistence import save_optimization_result
from app.optimization.report import build_markdown_report


class OptimizationPersistenceTests(unittest.TestCase):
    def _result(self) -> OptimizationResult:
        pm = PeriodMetrics(10, 0.1, 1.2, 5.0, 0.6, 55.0, 1.7, 2.0)
        ev = OptimizationEvaluation(
            "abc", "Classic",
            {"min_rr": 1.5, "max_rr": 2.0},
            pm, pm, pm,
            0.2, 0.2, 0.2,
            True, True, (),
            RobustnessSummary(0.9, 2, 0.1, 0.9, 1.0, 0.75, ()),
            {},
        )
        return OptimizationResult(
            "phase07-v1", "run1", datetime(2026, 1, 1, tzinfo=timezone.utc),
            "Classic",
            OptimizationCandidate("Classic", {"min_rr": 1.5}, research_reason="test").to_dict(),
            OptimizationSpace((ParameterSpec("min_rr", (1.5,), "test"),)),
            OptimizationConfig(min_validation_trades=0, min_oos_trades=0),
            SplitPlan(60, 80, 100),
            (ev,), (ev,), {},
        )

    def test_outputs_are_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_optimization_result(self._result(), tmp)
            self.assertEqual(len(paths), 4)
            self.assertIn("Phase 07", build_markdown_report(self._result()))
            self.assertTrue(all(Path(path).exists() for path in paths.values()))

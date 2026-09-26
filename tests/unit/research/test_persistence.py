from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.research.models import ResearchExperiment, ResearchMatrix, ResearchRun, ResearchScreeningConfig, default_variants, ExperimentAssessment, MetricSummary, SignalStats
from app.research.persistence import comparison_rows, save_research_run
from app.research.report import build_markdown_report, save_markdown_report


class PersistenceTests(unittest.TestCase):
    def _sample_run(self) -> ResearchRun:
        zero = MetricSummary(
            total_trades=0, wins=0, losses=0, expired=0, ambiguous_skipped=0,
            win_rate=0.0, profit_factor=None, expectancy=0.0, expectancy_r=0.0,
            average_rr=None, max_drawdown=0.0, max_drawdown_pct=0.0, average_trade=0.0,
            net_return=0.0, tp_rate=0.0, sl_rate=0.0, expiry_rate=0.0, return_stddev=0.0,
            win_rate_stddev_by_period=0.0, positive_period_ratio=0.0,
        )
        signals = SignalStats(10, 0, 10, 0, 0, 0.0, None)
        exp = ResearchExperiment(
            experiment_id="abc123", symbol="XAUUSD", timeframe="M15",
            strategy_name="Classic", strategy_variant="Classic_V1",
            variant_name=default_variants()[0].name,
            variant_config=dict(default_variants()[0].strategy_config.__dict__),
            is_metrics=zero, oos_metrics=zero, is_signal_stats=signals, oos_signal_stats=signals,
            assessment=ExperimentAssessment(("INSUFFICIENT_SAMPLE",), False, ("not enough trades",)),
            metadata={},
        )
        return ResearchRun(
            research_engine_version="phase06-v1", run_id="run123",
            generated_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            matrix=ResearchMatrix(timeframes=("M15",), variants=(default_variants()[0],)),
            screening=ResearchScreeningConfig(), experiments=(exp,), dataset_metadata={},
        )

    def test_machine_outputs_are_saved(self) -> None:
        run = self._sample_run()
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_research_run(run, tmp)
            report = save_markdown_report(run, Path(tmp) / "research_report.md")
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            self.assertTrue(report.exists())
            self.assertIn("Phase 06 Research Report", build_markdown_report(run))

    def test_comparison_rows_are_deterministic(self) -> None:
        rows = list(comparison_rows(self._sample_run().experiments))
        self.assertEqual(rows[0]["experiment_id"], "abc123")
        self.assertFalse(rows[0]["candidate_for_phase07"])

from __future__ import annotations

import unittest

from app.signals.calibration import build_calibration_report, monotonicity_violations


class CalibrationTests(unittest.TestCase):
    def test_calibration_report_has_consistent_sample_count_and_brier_score(self) -> None:
        report = build_calibration_report([20, 40, 60, 80], [0, 0, 1, 1], bins=4)
        self.assertEqual(report.sample_count, 4)
        self.assertIsNotNone(report.brier_score)
        self.assertEqual(sum(item.count for item in report.bins), 4)

    def test_monotonicity_detects_controlled_decrease(self) -> None:
        self.assertEqual(monotonicity_violations([0.1, 0.2, 0.3], [10, 20, 15]), 1)

    def test_monotonicity_is_clean_for_non_decreasing_confidence(self) -> None:
        self.assertEqual(monotonicity_violations([0.1, 0.2, 0.3], [10, 20, 20]), 0)


    def test_report_can_check_monotonicity_when_evidence_is_supplied(self) -> None:
        report = build_calibration_report(
            [30, 50, 40],
            [0, 1, 0],
            bins=3,
            evidence_scores=[0.1, 0.2, 0.3],
        )
        self.assertGreaterEqual(report.monotonicity_violations, 1)

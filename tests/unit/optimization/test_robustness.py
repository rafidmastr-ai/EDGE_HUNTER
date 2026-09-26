from __future__ import annotations

import unittest

from app.optimization.robustness import (
    RobustnessCase,
    cross_case_summary,
    detect_parameter_instability,
    summarize_robustness_flags,
)


class RobustnessTests(unittest.TestCase):
    def test_cross_case_ratio_is_correct(self) -> None:
        summary = cross_case_summary([
            RobustnessCase("A", {}, True),
            RobustnessCase("B", {}, False),
        ])
        self.assertAlmostEqual(summary["positive_case_ratio"], 0.5)

    def test_parameter_instability_flag(self) -> None:
        score, flags = detect_parameter_instability(0.2, 0.8, max_local_deterioration=0.5)
        self.assertEqual(score, 0.2)
        self.assertIn("PARAMETER_SENSITIVE", flags)

    def test_combined_flags_are_unique(self) -> None:
        flags = summarize_robustness_flags(
            positive_case_ratio=0.2,
            walk_forward_positive_fold_ratio=0.2,
            sensitivity_flags=("PARAMETER_SENSITIVE", "PARAMETER_SENSITIVE"),
        )
        self.assertEqual(flags.count("PARAMETER_SENSITIVE"), 1)
        self.assertIn("CROSS_CASE_UNSTABLE", flags)
        self.assertIn("WALK_FORWARD_UNSTABLE", flags)

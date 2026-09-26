from __future__ import annotations

import unittest

from app.research.models import ResearchMatrix, ResearchScreeningConfig, default_variants


class ResearchModelTests(unittest.TestCase):
    def test_default_matrix_has_expected_initial_families(self) -> None:
        matrix = ResearchMatrix.default()
        self.assertEqual(matrix.strategy_names, ("Classic", "SMC", "ICT"))
        self.assertEqual(matrix.timeframes, ("M1", "M5", "M15", "M30", "H1", "H4"))
        self.assertGreaterEqual(len(matrix.variants), 3)
        self.assertEqual(matrix.experiment_count(), 2 * 6 * 3 * len(matrix.variants))

    def test_variants_cover_requested_rr_range(self) -> None:
        names = {variant.name for variant in default_variants()}
        self.assertIn("RR_FIXED_1P50", names)
        self.assertIn("RR_FIXED_1P75", names)
        self.assertIn("RR_FIXED_2P00", names)
        self.assertEqual(default_variants()[0].strategy_config.min_rr, 1.5)
        self.assertEqual(default_variants()[0].strategy_config.max_rr, 2.0)

    def test_screening_config_rejects_invalid_ratio(self) -> None:
        with self.assertRaises(ValueError):
            ResearchScreeningConfig(minimum_oos_positive_period_ratio=1.1)

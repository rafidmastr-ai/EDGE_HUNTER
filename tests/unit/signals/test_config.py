from __future__ import annotations

import unittest

from app.signals.config import ConfidenceConfig


class ConfidenceConfigTests(unittest.TestCase):
    def test_default_weights_sum_to_one_and_labels_are_configurable(self) -> None:
        config = ConfidenceConfig()
        self.assertAlmostEqual(sum(config.weights().values()), 1.0)
        self.assertEqual(config.label_for(20), "ضعيف")
        self.assertEqual(config.label_for(50), "متوسط")
        self.assertEqual(config.label_for(70), "قوي")
        self.assertEqual(config.label_for(95), "قوي جدًا")

    def test_invalid_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConfidenceConfig(w_signal_quality=0.50)

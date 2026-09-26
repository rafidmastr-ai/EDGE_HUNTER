from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.optimization.sensitivity import compute_sensitivity


class SensitivityTests(unittest.TestCase):
    def test_local_deterioration_is_detected(self) -> None:
        items = [
            SimpleNamespace(evaluation_id="a", config={"x": 1, "y": 1}, validation_score=1.0),
            SimpleNamespace(evaluation_id="b", config={"x": 2, "y": 1}, validation_score=0.2),
            SimpleNamespace(evaluation_id="c", config={"x": 1, "y": 2}, validation_score=0.9),
        ]
        report = compute_sensitivity(items)
        self.assertEqual(report["a"]["neighbor_count"], 2)
        self.assertGreater(report["a"]["worst_neighbor_deterioration"], 0.7)

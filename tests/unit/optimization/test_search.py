from __future__ import annotations

import unittest

from app.optimization.models import OptimizationSpace, ParameterSpec
from app.optimization.search import generate_search_configs, sample_random


class OptimizationSearchTests(unittest.TestCase):
    def _space(self, method: str) -> OptimizationSpace:
        return OptimizationSpace(
            parameters=(
                ParameterSpec("min_rr", (1.5, 2.0), "Phase 06 R:R dimension."),
                ParameterSpec("max_rr", (1.5, 2.0), "Phase 06 R:R dimension."),
            ),
            method=method,
            max_trials=10,
            seed=7,
        )

    def test_grid_filters_invalid_rr_combinations(self) -> None:
        configs = generate_search_configs(self._space("grid"))
        self.assertEqual(len(configs), 3)
        self.assertTrue(all(item["min_rr"] <= item["max_rr"] for item in configs))

    def test_random_search_is_seed_deterministic(self) -> None:
        first = sample_random(self._space("random"))
        second = sample_random(self._space("random"))
        self.assertEqual(first, second)

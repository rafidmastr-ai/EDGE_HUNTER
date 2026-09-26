from __future__ import annotations

import unittest

from app.optimization.splits import make_split_plan, make_walk_forward_folds


class OptimizationSplitTests(unittest.TestCase):
    def test_train_validation_oos_are_strictly_chronological(self) -> None:
        split = make_split_plan(100, train_fraction=0.6, validation_fraction=0.2)
        self.assertLess(split.train_end, split.validation_end)
        self.assertLess(split.validation_end, split.total_bars)

    def test_walk_forward_windows_do_not_overlap_future_into_train(self) -> None:
        folds = make_walk_forward_folds(120, folds=3)
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(fold.train_end, fold.test_start)
            self.assertLess(fold.test_start, fold.test_end)

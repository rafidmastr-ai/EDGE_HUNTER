from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.optimization.models import OptimizationCandidate, OptimizationConfig, OptimizationSpace, ParameterSpec
from app.optimization.runner import OptimizationRunner
from app.optimization.splits import make_split_plan


def make_bars(count: int = 240):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    out = []
    for i in range(count):
        move = 0.25 if (i // 12) % 2 == 0 else -0.18
        open_ = price
        close = price + move
        high = max(open_, close) + 0.08
        low = min(open_, close) - 0.08
        out.append(
            __import__("app.data.schema", fromlist=["CanonicalOHLC"]).CanonicalOHLC(
                timestamp=base + timedelta(minutes=i),
                open=Decimal(str(round(open_, 5))),
                high=Decimal(str(round(high, 5))),
                low=Decimal(str(round(low, 5))),
                close=Decimal(str(round(close, 5))),
            )
        )
        price = close
    return tuple(out)


class OptimizationRunnerIntegrationTests(unittest.TestCase):
    def _runner(self):
        return OptimizationRunner(
            optimization_config=OptimizationConfig(
                min_validation_trades=0,
                min_oos_trades=0,
                top_train_pool_size=4,
                candidate_pool_size=2,
                walk_forward_folds=2,
            )
        )

    def _space(self):
        return OptimizationSpace(
            method="grid",
            max_trials=6,
            seed=11,
            parameters=(
                ParameterSpec("min_rr", (1.5, 2.0), "Phase 06 R:R evidence."),
                ParameterSpec("max_rr", (1.5, 2.0), "Phase 06 R:R evidence."),
            ),
        )

    def _candidate(self):
        return OptimizationCandidate(
            "Classic",
            {
                "min_rr": 1.5,
                "max_rr": 2.0,
                "swing_lookback": 5,
                "minimum_body_ratio": 0.55,
                "minimum_confirmation_score": 2.0,
            },
            research_reason="Phase 06 R:R evidence.",
        )

    def test_optimizer_keeps_oos_separate_and_records_metadata(self):
        result = self._runner().optimize(
            self._candidate(),
            make_bars(),
            symbol="XAUUSD",
            timeframe="M1",
            space=self._space(),
        )
        split = make_split_plan(240)
        self.assertEqual(result.split.to_dict(), split.to_dict())
        self.assertEqual(len(result.evaluations), 3)  # 3 valid RR combinations
        self.assertTrue(all("dataset_sha256" not in item.metadata for item in result.evaluations))
        self.assertTrue(result.run_id)
        self.assertEqual(result.space.seed, 11)

    def test_same_inputs_produce_same_run_id(self):
        runner = self._runner()
        first = runner.optimize(self._candidate(), make_bars(), symbol="XAUUSD", timeframe="M1", space=self._space())
        second = runner.optimize(self._candidate(), make_bars(), symbol="XAUUSD", timeframe="M1", space=self._space())
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual([e.to_dict() for e in first.evaluations], [e.to_dict() for e in second.evaluations])

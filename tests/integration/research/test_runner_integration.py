from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.research.models import ResearchMatrix, ResearchScreeningConfig
from app.research.runner import ResearchRunner


def make_bars(count: int = 120) -> tuple[CanonicalOHLC, ...]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    output = []
    price = 100.0
    for index in range(count):
        move = 0.2 if (index // 8) % 2 == 0 else -0.15
        open_ = price
        close = price + move
        high = max(open_, close) + 0.08
        low = min(open_, close) - 0.08
        output.append(
            CanonicalOHLC(
                timestamp=base + timedelta(minutes=index),
                open=Decimal(str(round(open_, 5))),
                high=Decimal(str(round(high, 5))),
                low=Decimal(str(round(low, 5))),
                close=Decimal(str(round(close, 5))),
            )
        )
        price = close
    return tuple(output)


class ResearchRunnerIntegrationTests(unittest.TestCase):
    def test_runner_executes_initial_strategy_families_through_common_engine(self) -> None:
        matrix = ResearchMatrix(
            symbols=("XAUUSD",),
            timeframes=("M1", "M15"),
            strategy_names=("Classic", "SMC", "ICT"),
            variants=(ResearchMatrix.quick().variants[0],),
            is_fraction=0.7,
        )
        run = ResearchRunner(screening=ResearchScreeningConfig(minimum_is_trades=0, minimum_oos_trades=0)).run_dataset(
            {"XAUUSD": make_bars()}, matrix
        )
        self.assertEqual(len(run.experiments), 6)
        self.assertEqual({e.strategy_name for e in run.experiments}, {"Classic", "SMC", "ICT"})
        self.assertTrue(all(e.experiment_id for e in run.experiments))

    def test_research_run_is_reproducible_for_same_inputs(self) -> None:
        matrix = ResearchMatrix(
            symbols=("XAUUSD",),
            timeframes=("M1",),
            strategy_names=("Classic",),
            variants=(ResearchMatrix.quick().variants[0],),
        )
        runner = ResearchRunner(screening=ResearchScreeningConfig(minimum_is_trades=0, minimum_oos_trades=0))
        first = runner.run_dataset({"XAUUSD": make_bars()}, matrix)
        second = runner.run_dataset({"XAUUSD": make_bars()}, matrix)
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.experiments[0].to_dict(), second.experiments[0].to_dict())

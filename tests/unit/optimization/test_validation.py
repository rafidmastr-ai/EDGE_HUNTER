from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.backtest.config import BacktestConfig
from app.features.engine import FeatureEngine
from app.optimization.validation import evaluate_three_periods, walk_forward_validate
from app.strategies.models import StrategyConfig
from app.strategies.models import StrategyConfig as SC
from app.data.schema import CanonicalOHLC


def make_bars(count=120):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    output=[]
    for i in range(count):
        move=0.1 if i % 2 == 0 else -0.05
        o=price
        c=price+move
        output.append(CanonicalOHLC(
            timestamp=base+timedelta(minutes=i),
            open=Decimal(str(o)),
            high=Decimal(str(max(o,c)+0.03)),
            low=Decimal(str(min(o,c)-0.03)),
            close=Decimal(str(c)),
        ))
        price=c
    return tuple(output)


class ValidationTests(unittest.TestCase):
    def test_three_periods_are_generated(self):
        bars=make_bars()
        analysis=FeatureEngine().compute(bars,symbol="XAUUSD",timeframe="M1")
        train,val,oos=evaluate_three_periods(
            "Classic", StrategyConfig(), bars=bars, analysis=analysis,
            backtest_config=BacktestConfig(), train_end=72, validation_end=96,
            symbol="XAUUSD", timeframe="M1"
        )
        self.assertIsNotNone(train)
        self.assertIsNotNone(val)
        self.assertIsNotNone(oos)

    def test_walk_forward_is_deterministic_structure(self):
        bars=make_bars()
        analysis=FeatureEngine().compute(bars,symbol="XAUUSD",timeframe="M1")
        summary=walk_forward_validate(
            "Classic", StrategyConfig(), bars=bars, analysis=analysis,
            backtest_config=BacktestConfig(), symbol="XAUUSD", timeframe="M1", folds=2
        )
        self.assertEqual(len(summary.folds), 2)
        for fold in summary.folds:
            self.assertLess(fold.train_end, fold.test_end)

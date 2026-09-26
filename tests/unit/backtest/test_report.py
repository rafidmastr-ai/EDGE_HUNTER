from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.backtest.engine import BacktestEngine
from app.backtest.report import save_json, to_json
from app.data.schema import CanonicalOHLC
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from app.strategies.models import SignalDirection, SignalState, StrategySignal


class OneSignalStrategy:
    name = "ReportTest"
    variant = "ReportTest_V1"

    def generate(self, context):
        if context.decision_index != 0:
            return StrategySignal(
                timestamp=context.current.timestamp,
                symbol=context.symbol,
                timeframe=context.timeframe,
                direction=SignalDirection.NO_SIGNAL,
                state=SignalState.NO_SIGNAL,
                entry=None,
                stop_loss=None,
                target=None,
                risk_reward=None,
                entry_logic="none",
                invalidation="none",
                stop_loss_logic="none",
                target_logic="none",
                evidence=("none",),
                strategy_name=self.name,
                variant=self.variant,
            )
        return StrategySignal(
            timestamp=context.current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=SignalDirection.BUY,
            state=SignalState.SIGNAL,
            entry=100.0,
            stop_loss=98.0,
            target=102.0,
            risk_reward=1.0,
            entry_logic="test",
            invalidation="test",
            stop_loss_logic="test",
            target_logic="test",
            evidence=("test",),
            strategy_name=self.name,
            variant=self.variant,
        )


def make_bars():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        CanonicalOHLC(base, Decimal("100"), Decimal("100.5"), Decimal("99.5"), Decimal("100")),
        CanonicalOHLC(base + timedelta(minutes=1), Decimal("100"), Decimal("103"), Decimal("99.5"), Decimal("101")),
    )


class ReportTests(unittest.TestCase):
    def test_result_is_json_serializable(self) -> None:
        result = BacktestEngine().run(OneSignalStrategy(), make_bars(), symbol="XAUUSD", timeframe="M1")
        payload = json.loads(to_json(result))
        self.assertEqual(payload["engine_version"], "phase05-v1")
        self.assertEqual(len(payload["trades"]), 1)

    def test_report_can_be_saved(self) -> None:
        result = BacktestEngine().run(OneSignalStrategy(), make_bars(), symbol="XAUUSD", timeframe="M1")
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report.json"
            saved = save_json(result, destination)
            self.assertTrue(saved.exists())
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["symbol"], "XAUUSD")

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.backtest.models import TradeOutcome, TradeRecord
from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from app.research.regimes import summarize_trade_regimes


class RegimeSummaryTests(unittest.TestCase):
    def test_trade_regime_uses_decision_time_information_only(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = tuple(
            CanonicalOHLC(
                timestamp=base + timedelta(minutes=i),
                open=Decimal(str(100 + i * 0.01)),
                high=Decimal(str(100 + i * 0.01 + 0.5)),
                low=Decimal(str(100 + i * 0.01 - 0.5)),
                close=Decimal(str(100 + i * 0.01)),
            )
            for i in range(210)
        )
        analysis = FeatureEngine().compute(bars, "XAUUSD", "M1")
        future_ignored_trade = TradeRecord(
            strategy_name="Synthetic", strategy_variant="Synthetic_V1", symbol="XAUUSD", timeframe="M1",
            direction="BUY", signal_timestamp=bars[200].timestamp, entry_timestamp=bars[200].timestamp,
            entry_price=100.0, stop_loss=99.0, target=101.0, planned_risk_reward=1.0, quantity=1.0,
            exit_timestamp=bars[201].timestamp, exit_price=101.0, outcome=TradeOutcome.WIN, r_multiple=1.0,
            pnl_gross=1.0, commission=0.0, pnl_net=1.0, bars_held=1, exit_reason="TP",
        )
        result_before = summarize_trade_regimes((future_ignored_trade,), analysis)
        extended = bars + tuple(
            CanonicalOHLC(
                timestamp=base + timedelta(minutes=i),
                open=Decimal("200"), high=Decimal("300"), low=Decimal("100"), close=Decimal("250"),
            ) for i in range(210, 220)
        )
        analysis_after = FeatureEngine().compute(extended, "XAUUSD", "M1")
        result_after = summarize_trade_regimes((future_ignored_trade,), analysis_after)
        self.assertEqual(result_before, result_after)

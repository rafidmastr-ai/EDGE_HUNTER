from __future__ import annotations

import unittest

from app.backtest.config import BacktestConfig, IntrabarPolicy


class BacktestConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = BacktestConfig()
        self.assertEqual(config.intrabar_policy, IntrabarPolicy.STOP_FIRST)
        self.assertGreater(config.starting_capital or 0, 0)

    def test_risk_based_position_sizing(self) -> None:
        config = BacktestConfig(starting_capital=10_000.0, risk_per_trade_pct=1.0)
        units = config.position_units(100.0, 98.0, capital=10_000.0)
        self.assertAlmostEqual(units, 50.0)

    def test_manual_lot_converts_to_units(self) -> None:
        config = BacktestConfig(lot_size=0.5, contract_size=100_000)
        self.assertEqual(config.position_units(1.0, 0.99), 50_000)

    def test_conflicting_sizing_modes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig(risk_per_trade_pct=1.0, lot_size=0.1)

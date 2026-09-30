from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.web.analysis_service import DataUnavailableError, LocalOHLCAnalysisService
from app.web.schemas import AnalyzeRequest


class AnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path = self.root / "XAUUSD.csv"
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close\n")
            price = 2300.0
            for i in range(720):
                timestamp = start + timedelta(minutes=i)
                open_price = price
                close_price = price + (0.25 if i % 3 else -0.1)
                high = max(open_price, close_price) + 0.45
                low = min(open_price, close_price) - 0.4
                handle.write(f"{timestamp.isoformat()},{open_price:.3f},{high:.3f},{low:.3f},{close_price:.3f}\n")
                price = close_price

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_service_returns_structured_result(self) -> None:
        service = LocalOHLCAnalysisService(self.root)
        result = service.analyze(AnalyzeRequest(symbol="XAUUSD", risk_percent=1.0, lot_mode="auto"))
        self.assertEqual(result["symbol"], "XAUUSD")
        self.assertIn(result["direction"], {"BUY", "SELL", "NO_CLEAR_SIGNAL"})
        self.assertEqual(len(result["analyzed_timeframes"]), 3)
        self.assertTrue(result["chart"])
        self.assertNotIn("capital_impact", {})
        self.assertIsNone(result["capital_impact"])

    def test_capital_impact_only_when_capital_is_supplied(self) -> None:
        service = LocalOHLCAnalysisService(self.root)
        without = service.analyze(AnalyzeRequest(symbol="XAUUSD", capital=None))
        self.assertIsNone(without["capital_impact"])
        with_capital = service.analyze(AnalyzeRequest(symbol="XAUUSD", capital=10000))
        self.assertIn("capital_impact", with_capital)

    def test_edge_ml_status_never_changes_the_main_signal(self) -> None:
        class StubEdgeML:
            def status(self, symbol):
                return {"enabled": True, "experimental": True, "models": [{"model": f"B1/{symbol}"}]}

        class BrokenEdgeML:
            def status(self, symbol):
                raise RuntimeError("boom")

        request = AnalyzeRequest(symbol="XAUUSD", risk_percent=1.0, lot_mode="auto")
        plain = LocalOHLCAnalysisService(self.root).analyze(request)
        with_edge = LocalOHLCAnalysisService(self.root, edge_ml=StubEdgeML()).analyze(request)
        broken = LocalOHLCAnalysisService(self.root, edge_ml=BrokenEdgeML()).analyze(request)
        self.assertEqual(plain["metadata"]["edge_ml"], {"enabled": False})
        self.assertEqual(with_edge["metadata"]["edge_ml"]["models"], [{"model": "B1/XAUUSD"}])
        self.assertEqual(broken["metadata"]["edge_ml"]["last_error"], "edge_ml_status_failed")
        for key in ("direction", "confidence", "entry", "target", "stop_loss", "selected_strategy", "reasons"):
            self.assertEqual(with_edge[key], plain[key])
            self.assertEqual(broken[key], plain[key])

    def test_missing_symbol_is_data_unavailable(self) -> None:
        service = LocalOHLCAnalysisService(self.root)
        with self.assertRaises(DataUnavailableError):
            service.analyze(AnalyzeRequest(symbol="EURUSD"))

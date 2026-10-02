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

    @staticmethod
    def _recommendation(direction: str = "SELL") -> dict:
        d = 1 if direction == "BUY" else -1
        return {
            "symbol": "XAUUSD", "model": "B1/XAUUSD", "variant": "B1", "variant_title_ar": "حتى الهدف/الوقف",
            "exit_rule_ar": "اخرج عند الهدف أو الوقف، أو بعد 12 ساعة", "direction": direction, "entry": 2400.0,
            "stop_loss": 2400.0 - d * 10.0, "take_profit": 2400.0 + d * 10.0, "risk_reward": 1.0, "atr_m15": 5.0,
            "bar_close_utc": "2026-09-29T10:00:00+00:00", "age_minutes": 3.0, "forecast_r": -0.3 * d, "expected_r": 0.29,
            "research_results": {"new_forward": {"trades": 110, "win_rate": 0.46, "avg_r": -0.06, "profit_factor": 0.88}},
            "costs": {"spread": 0.1, "swap_per_night": 0.5},
        }

    def _stub(self, recommendation=None, ranked=None, closest="XAUUSD"):
        rec = recommendation

        class Stub:
            def status(self, symbol):
                return {"enabled": True, "experimental": True, "symbol": symbol, "models": []}

            def recommendation(self, symbol):
                return rec if rec and rec["symbol"] == symbol.replace("/", "") else None

            def best_recommendation(self):
                return rec, (ranked if ranked is not None else ([rec] if rec else []))

            def closest_symbol(self):
                return closest

            def quote_to_usd(self, symbol):
                return 1.0

        return Stub()

    def test_active_model_signal_becomes_the_main_result(self) -> None:
        request = AnalyzeRequest(symbol="XAUUSD", risk_percent=1.0, capital=10_000, lot_mode="auto")
        plain = LocalOHLCAnalysisService(self.root).analyze(request)
        result = LocalOHLCAnalysisService(self.root, edge_ml=self._stub(self._recommendation("SELL"))).analyze(request)
        self.assertEqual(result["direction"], "SELL")
        self.assertEqual((result["entry"], result["stop_loss"], result["target"]), (2400.0, 2410.0, 2390.0))
        self.assertEqual(result["selected_strategy"], "EDGE ML B1")
        self.assertEqual(result["metadata"]["recommendation_source"], "edge_ml")
        self.assertEqual(result["metadata"]["classic_analysis"]["direction"], plain["direction"])
        # 1 % of 10,000 = 100 USD at a 10 USD stop on 100 oz per lot -> 0.10 lot
        self.assertEqual(result["lot_size"], 0.1)
        self.assertEqual(result["capital_impact"]["stop_loss_loss"], 100.0)
        self.assertEqual(result["chart"], plain["chart"])

    def test_no_model_signal_keeps_the_classic_result(self) -> None:
        request = AnalyzeRequest(symbol="XAUUSD", risk_percent=1.0, lot_mode="auto")
        plain = LocalOHLCAnalysisService(self.root).analyze(request)
        result = LocalOHLCAnalysisService(self.root, edge_ml=self._stub(None)).analyze(request)
        self.assertEqual(result["metadata"]["recommendation_source"], "classic")
        for key in ("direction", "confidence", "entry", "target", "stop_loss", "selected_strategy", "reasons", "lot_size"):
            self.assertEqual(result[key], plain[key])

    def test_empty_symbol_picks_the_best_model_recommendation(self) -> None:
        request = AnalyzeRequest(symbol="", risk_percent=1.0, lot_mode="auto")
        self.assertIsNone(request.symbol)
        result = LocalOHLCAnalysisService(self.root, edge_ml=self._stub(self._recommendation("BUY"))).analyze(request)
        self.assertEqual(result["symbol"], "XAUUSD")
        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["metadata"]["recommendation_source"], "edge_ml_auto")
        self.assertEqual(result["metadata"]["edge_ml_ranked"][0]["model"], "B1/XAUUSD")
        self.assertIsNone(result["metadata"]["edge_ml"]["symbol"])  # card shows every model

    def test_empty_symbol_without_any_signal_is_no_clear_signal(self) -> None:
        request = AnalyzeRequest(symbol=None, risk_percent=1.0, lot_mode="auto")
        result = LocalOHLCAnalysisService(self.root, edge_ml=self._stub(None, closest="XAUUSD")).analyze(request)
        self.assertEqual(result["symbol"], "XAUUSD")
        self.assertEqual(result["direction"], "NO_CLEAR_SIGNAL")
        self.assertEqual(result["status"], "no_clear_signal")
        self.assertIsNone(result["entry"])
        self.assertIn("لا توجد توصية", result["reasons"][0])
        self.assertTrue(result["chart"])

    def test_below_threshold_pick_is_labelled_weak(self) -> None:
        rec = {**self._recommendation("BUY"), "tier": "below_threshold"}
        result = LocalOHLCAnalysisService(self.root, edge_ml=self._stub(rec)).analyze(AnalyzeRequest(symbol=None))
        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["confidence_label_ar"], "ضعيف")
        self.assertLessEqual(result["confidence"], 25.0)
        self.assertIn("دون عتبة الدخول المختبرة", result["reasons"][0])
        self.assertEqual(result["metadata"]["recommendation_tier"], "below_threshold")

    def test_no_recommendation_explains_why(self) -> None:
        stub = self._stub(None)
        stub.readiness = lambda: {"problems_ar": ["النماذج قيد التحميل والتحديث الأول"]}
        result = LocalOHLCAnalysisService(self.root, edge_ml=stub).analyze(AnalyzeRequest(symbol=None))
        self.assertIn("النماذج قيد التحميل والتحديث الأول", result["reasons"])

    def test_classic_no_signal_on_a_pair_without_a_model_points_to_the_model_pairs(self) -> None:
        stub = self._stub(None)
        stub.model_symbols = lambda: ["AUDJPY", "EURJPY"]
        stub.readiness = lambda: {"problems_ar": []}
        service = LocalOHLCAnalysisService(self.root, edge_ml=stub)
        service._analyze_classic_original = service._analyze_classic

        def classic(request):
            result, obs = service._analyze_classic_original(request)
            result["direction"] = "NO_CLEAR_SIGNAL"
            return result, obs

        service._analyze_classic = classic
        result = service.analyze(AnalyzeRequest(symbol="XAUUSD"))
        self.assertTrue(any("AUDJPY, EURJPY" in r for r in result["reasons"]))
        self.assertTrue(any("فارغاً" in r for r in result["reasons"]))

    def test_empty_symbol_needs_edge_ml(self) -> None:
        with self.assertRaises(ValueError):
            LocalOHLCAnalysisService(self.root).analyze(AnalyzeRequest(symbol=None))

    def test_invalid_symbol_is_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnalyzeRequest(symbol="$$")

    def test_missing_symbol_is_data_unavailable(self) -> None:
        service = LocalOHLCAnalysisService(self.root)
        with self.assertRaises(DataUnavailableError):
            service.analyze(AnalyzeRequest(symbol="EURUSD"))

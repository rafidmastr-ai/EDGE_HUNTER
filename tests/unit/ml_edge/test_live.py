from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.ml_edge.data import SymbolData, cost_price
from app.ml_edge.features import build_all
from app.ml_edge.labels import LabelConfig
from app.ml_edge.live import SYMBOLS, EdgeMLService, LatestDecision, market_open, normalize_symbol
from app.ml_edge.live_store import M1Store, StoredSeries, fetch_updates
from app.ml_edge.model import SymbolModel
from app.ml_edge.walkforward import execution_config
from app.providers.models import LiveProviderError
from app.research.intraday import Bars, make_orders, simulate

START = 1_767_571_200  # 2026-01-05 00:00 UTC (Monday)
PRICES = {"AUDUSD": 0.66, "EURJPY": 160.0, "EURUSD": 1.08, "GBPUSD": 1.27, "NZDUSD": 0.60, "XAUUSD": 2400.0,
          "AUDJPY": 105.0, "CADJPY": 110.0}


def synthetic(symbol: str, days: int, seed: int) -> StoredSeries:
    rng = np.random.default_rng(seed)
    t = START + 60 * np.arange(days * 1440, dtype=np.int64)
    weekday = ((t // 86400) + 3) % 7
    t = t[weekday < 5]
    c = PRICES[symbol] * np.exp(np.cumsum(rng.normal(0, 2e-4, len(t))))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 1e-4, len(t))) * PRICES[symbol]
    return StoredSeries(t, o, np.maximum(o, c) + spread, np.minimum(o, c) - spread, c,
                        rng.integers(50, 150, len(t)).astype(float), rng.integers(3, 9, len(t)).astype(float))


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


class FakeProvider:
    """Returns flat minute bars for any range; optional scripted errors per call."""

    def __init__(self, errors: dict[int, str] | None = None) -> None:
        self.calls: list[tuple[datetime, datetime]] = []
        self.errors = errors or {}

    def get_ohlc(self, symbol, timeframe, start, end):
        self.calls.append((start, end))
        code = self.errors.get(len(self.calls))
        if code:
            raise LiveProviderError("scripted", code=code)
        bars, t = [], start
        while t <= end:
            bars.append(Bar(t, 1.0, 1.001, 0.999, 1.0005))
            t += timedelta(minutes=1)
        return bars


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_append_keeps_existing_bars_rejects_bad_ones_and_persists(self) -> None:
        store = M1Store(self.dir, ("EURUSD",))
        self.assertEqual(store.append("EURUSD", [(600, 1.0, 1.1, 0.9, 1.05), (660, 1.05, 1.06, 1.0, 1.01)]), 2)
        # 600 already stored (different prices are ignored); 720 is invalid (high < close)
        self.assertEqual(store.append("EURUSD", [(600, 2.0, 2.0, 2.0, 2.0), (720, 1.0, 1.0, 0.9, 1.2), (780, 1.0, 1.0, 1.0, 1.0)]), 1)
        series = M1Store(self.dir, ("EURUSD",)).series("EURUSD")  # reloaded from disk
        self.assertEqual(series.t.tolist(), [600, 660, 780])
        self.assertEqual(series.o[0], 1.0)
        self.assertTrue(np.isnan(series.v).all())  # live bars carry no tick volume

    def test_keeps_only_the_recent_window(self) -> None:
        store = M1Store(self.dir, ("EURUSD",), keep_days=1)
        store.append("EURUSD", [(0, 1, 1, 1, 1), (2 * 86400, 1, 1, 1, 1)])
        self.assertEqual(store.series("EURUSD").t.tolist(), [2 * 86400])

    def test_fetch_pages_from_the_last_bar_and_pauses_between_requests(self) -> None:
        store = M1Store(self.dir, ("EURUSD",))
        last = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        store.append("EURUSD", [(int(last.timestamp()), 1.0, 1.0, 1.0, 1.0)])
        provider, pauses = FakeProvider(), []
        now = last + timedelta(minutes=1500, seconds=30)
        added = fetch_updates(store, provider, "EURUSD", now, page_minutes=700, pause=pauses.append, pause_seconds=20)
        self.assertEqual(added, 1499)  # every closed minute after the stored bar, the running minute excluded
        self.assertEqual(provider.calls[0][0], last + timedelta(minutes=1))
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(pauses, [20, 20])
        self.assertEqual(store.series("EURUSD").last_time, int((now - timedelta(seconds=30, minutes=1)).timestamp()))

    def test_fetch_skips_closed_market_windows_and_stops_on_other_errors(self) -> None:
        store = M1Store(self.dir, ("EURUSD",))
        last = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        store.append("EURUSD", [(int(last.timestamp()), 1.0, 1.0, 1.0, 1.0)])
        provider = FakeProvider({1: "live_empty_data", 2: "provider_http_500"})
        added = fetch_updates(store, provider, "EURUSD", last + timedelta(minutes=2000), page_minutes=700, pause=lambda s: None)
        self.assertEqual(added, 0)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[1][0], provider.calls[0][1] + timedelta(minutes=1))

    def test_shared_rate_limit_waits_and_retries_instead_of_giving_up(self) -> None:
        store = M1Store(self.dir, ("EURUSD",))
        last = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        store.append("EURUSD", [(int(last.timestamp()), 1.0, 1.0, 1.0, 1.0)])
        provider, pauses = FakeProvider({2: "provider_local_rate_limited", 3: "provider_local_rate_limited"}), []
        added = fetch_updates(store, provider, "EURUSD", last + timedelta(minutes=1500), page_minutes=700, pause=pauses.append)
        self.assertEqual(added, 1499)  # every page was eventually fetched
        self.assertEqual(pauses.count(20.0), 2)  # two waits for the rate limit

    def test_fetch_leaves_room_for_user_analyses(self) -> None:
        import time as _time
        from collections import deque

        store = M1Store(self.dir, ("EURUSD",))
        last = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
        store.append("EURUSD", [(int(last.timestamp()), 1.0, 1.0, 1.0, 1.0)])
        provider, pauses = FakeProvider(), []
        provider._rate_limit = 8
        provider._request_times = deque([_time.monotonic()] * 5)  # 5 recent requests: only 3 left

        def pause(seconds):
            pauses.append(seconds)
            provider._request_times.clear()  # the minute passes

        fetch_updates(store, provider, "EURUSD", last + timedelta(minutes=30), pause=pause)
        self.assertEqual(pauses, [5.0])  # waited once before using the reserved slots

    def test_twelve_data_no_data_answer_skips_the_window(self) -> None:
        # a closed market (weekend) answers {"code": 400, "message": "No data is available ..."}
        store = M1Store(self.dir, ("EURUSD",))
        friday = datetime(2026, 1, 9, 20, 59, tzinfo=timezone.utc)
        store.append("EURUSD", [(int(friday.timestamp()), 1.0, 1.0, 1.0, 1.0)])
        provider = FakeProvider({1: "provider_400", 2: "provider_400", 3: "provider_400", 4: "provider_400"})
        monday = datetime(2026, 1, 12, 9, 0, tzinfo=timezone.utc)
        added = fetch_updates(store, provider, "EURUSD", monday, pause=lambda s: None, max_pages=60)
        self.assertGreater(added, 0)  # the store moved past the weekend instead of stopping there
        self.assertEqual(store.series("EURUSD").last_time, int((monday - timedelta(minutes=1)).timestamp()))

    def test_market_hours(self) -> None:
        self.assertFalse(market_open(datetime(2026, 1, 10, 12, tzinfo=timezone.utc)))  # Saturday
        self.assertFalse(market_open(datetime(2026, 1, 11, 20, tzinfo=timezone.utc)))  # Sunday before open
        self.assertTrue(market_open(datetime(2026, 1, 11, 22, tzinfo=timezone.utc)))
        self.assertFalse(market_open(datetime(2026, 1, 9, 21, 30, tzinfo=timezone.utc)))  # Friday after close
        self.assertTrue(market_open(datetime(2026, 1, 7, 3, tzinfo=timezone.utc)))


class ServiceTests(unittest.TestCase):
    DAYS = 100

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.series = {s: synthetic(s, cls.DAYS, k) for k, s in enumerate(SYMBOLS)}
        # the store ends at 12:00 UTC on the last weekday (a live store never holds future bars)
        cut = int(cls.series["EURJPY"].t[-1]) // 86400 * 86400 + 12 * 3600
        cls.series = {s: StoredSeries(*(getattr(v, f)[v.t < cut] for f in ("t", "o", "h", "l", "c", "v", "s"))) for s, v in cls.series.items()}
        cls.store_dir = root / "store"
        seed_store = M1Store(cls.store_dir, SYMBOLS)
        for s, series in cls.series.items():
            seed_store._save(s, series)
        frames = build_all({s: v.to_symbol_data(s) for s, v in cls.series.items()}, session_start_hour=1)
        frame = frames["EURJPY"]
        ok = frame.tradable
        rng = np.random.default_rng(3)
        y = rng.normal(0, 1, int(ok.sum()))
        cls.models = root / "models"
        label = LabelConfig(2.0, 48, max_hold_minutes=720)
        model = SymbolModel("EURJPY", "symbol", "test", label, frame.names, 0.05, True, cost_price("EURJPY"),
                            symbol_model=HistGradientBoostingRegressor(max_iter=20).fit(frame.x[ok], y), swap_per_night=0.005)
        model.save(cls.models / "B1" / "EURJPY")
        model.save(cls.models / "B1" / "GBPUSD")
        (cls.models / "B1" / "GBPUSD" / "model.joblib").write_bytes(b"tampered")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.db_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.db_dir.name) / "t.db")
        MigrationRunner(self.db).apply_all()

    def tearDown(self) -> None:
        self.db.close()
        self.db_dir.cleanup()

    def service(self) -> EdgeMLService:
        return EdgeMLService(self.models, M1Store(self.store_dir, SYMBOLS), self.db)

    def test_tampered_model_is_reported_not_loaded(self) -> None:
        status = self.service().status()
        self.assertEqual([m["model"] for m in status["models"]], ["B1/EURJPY"])
        self.assertIn("B1/GBPUSD", status["load_errors"])
        self.assertTrue(status["experimental"])

    def test_decisions_and_paper_trades_match_the_backtest_engine(self) -> None:
        svc = self.service()
        last = int(self.series["EURJPY"].t[-1])
        now = datetime.fromtimestamp(last + 60, tz=timezone.utc)
        svc._paper_start = last - 20 * 86400
        svc.refresh(now, fetch=False)
        self.assertIsNone(svc.state.last_error)
        latest = svc.status("EURJPY", now=now)["models"][0]["latest"]
        self.assertTrue(latest["fresh"])
        self.assertIn(latest["direction"], {"BUY", "SELL", "NONE"})

        # the same decisions through the research engine
        frame = build_all({s: v.to_symbol_data(s) for s, v in self.series.items()}, session_start_hour=1)["EURJPY"]
        x = frame.x.copy()
        keep = frame.tradable & (frame.close_time <= last + 60) & (frame.close_time >= last + 60 - 80 * 86400)
        model = svc.models[0].model
        direction, _ = model.decide(frame.close_time[keep], x[keep], frame.atr[keep])
        times, atr = frame.close_time[keep], frame.atr[keep]
        sel = (direction != 0) & (times >= last + 60 - 2 * 86400)
        n = int(sel.sum())
        orders = make_orders(times[sel], direction[sel], np.full(n, np.nan), np.full(n, np.nan),
                             stop_distance=2.0 * atr[sel], target_rr=np.ones(n))
        expected = simulate(self.series["EURJPY"].to_symbol_data("EURJPY").m1, orders, execution_config("EURJPY", model.label))
        rows = sorted(svc.paper_trades("EURJPY", 500), key=lambda r: r["signal_time"])
        self.assertGreater(len(expected), 0)
        self.assertEqual(len(rows), len(expected))
        for row, i in zip(rows, range(len(expected))):
            self.assertAlmostEqual(row["entry"], float(expected.entry[i]))
            if row["status"] == "closed":
                self.assertAlmostEqual(row["r_net"], float(expected.r_net[i]))

        # a second refresh is idempotent
        svc.refresh(now, fetch=False)
        self.assertEqual(len(svc.paper_trades("EURJPY", 500)), len(rows))

    def test_old_decision_is_stale(self) -> None:
        svc = self.service()
        last = int(self.series["EURJPY"].t[-1])
        later = datetime.fromtimestamp(last + 3 * 3600, tz=timezone.utc)
        svc.refresh(later, fetch=False)
        self.assertFalse(svc.status("EURJPY", now=later)["models"][0]["latest"]["fresh"])


class RecommendationTests(unittest.TestCase):
    MODELS = Path(__file__).resolve().parents[3] / "models" / "edge_ml"
    NOW = datetime(2026, 9, 29, 10, 5, tzinfo=timezone.utc)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.svc = EdgeMLService(self.MODELS, M1Store(Path(self.tmp.name), SYMBOLS))
        self.svc.ensure_loaded()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def decide(self, key: str, direction: str, expected_r: float, minutes_ago: int = 5, forecast: float = 0.3) -> None:
        bar = (self.NOW - timedelta(minutes=minutes_ago)).isoformat()
        d = 1 if direction == "BUY" else -1
        self.svc.state.decisions[key] = LatestDecision(bar, direction, forecast * d, 0.1, 150.0, 150.0 - d * 0.2,
                                                       150.0 + d * 0.2, True, expected_r)

    def test_all_enabled_models_and_yen_crosses_are_served(self) -> None:
        keys = {m.key for m in self.svc.models}
        self.assertEqual(keys, {"A/AUDUSD", "A/EURJPY", "A/GBPUSD", "A/NZDUSD", "B1/EURJPY", "B1/GBPUSD",
                                "B1/NZDUSD", "B1/XAUUSD", "B1/AUDJPY", "B1/CADJPY"})
        self.assertEqual(self.svc.load_errors, {})

    def test_best_recommendation_ranks_by_expected_r_and_ignores_stale_or_none(self) -> None:
        self.decide("B1/EURJPY", "SELL", 0.20)
        self.decide("B1/CADJPY", "BUY", 0.35)
        self.decide("A/GBPUSD", "BUY", 0.90, minutes_ago=60)  # stale: older than the refresh window
        self.svc.state.decisions["B1/XAUUSD"] = LatestDecision((self.NOW - timedelta(minutes=5)).isoformat(), "NONE", 0.5)
        best, ranked = self.svc.best_recommendation(self.NOW)
        self.assertEqual(best["symbol"], "CADJPY")
        self.assertEqual([r["model"] for r in ranked], ["B1/CADJPY", "B1/EURJPY"])
        self.assertEqual(best["risk_reward"], 1.0)
        self.assertIn("12 ساعة", best["exit_rule_ar"])
        self.assertIn("new_forward", best["research_results"])

    def test_without_active_signal_the_strongest_fresh_candidate_is_offered(self) -> None:
        bar = (self.NOW - timedelta(minutes=5)).isoformat()
        self.svc.state.decisions["B1/EURJPY"] = LatestDecision(bar, "NONE", -0.30, 0.1, 150.0, None, None, True, None,
                                                               "SELL", 150.2, 149.8, 0.29)
        self.svc.state.decisions["A/GBPUSD"] = LatestDecision(bar, "NONE", 0.10, 0.001, 1.3, None, None, True, None,
                                                              "BUY", 1.298, 1.302, 0.09)
        best, ranked = self.svc.best_recommendation(self.NOW)
        self.assertEqual((best["model"], best["tier"], best["direction"]), ("B1/EURJPY", "below_threshold", "SELL"))
        self.assertEqual((best["stop_loss"], best["take_profit"]), (150.2, 149.8))
        self.assertIsNone(self.svc.recommendation("EURJPY", self.NOW))  # a chosen symbol needs a real signal
        # an active signal always ranks above any candidate
        self.decide("A/NZDUSD", "BUY", 0.05)
        best, _ = self.svc.best_recommendation(self.NOW)
        self.assertEqual((best["model"], best["tier"]), ("A/NZDUSD", "active"))

    def test_readiness_explains_missing_history_and_warm_up(self) -> None:
        r = self.svc.readiness(self.NOW)
        self.assertFalse(r["ready"])
        self.assertTrue(any("مزوّد البيانات الحية غير مفعّل" in p for p in r["problems_ar"]))
        self.svc.state.store_coverage = {"AUDJPY": {"days": 4.0}, "EURJPY": {"days": 150.0}}
        self.assertTrue(any("AUDJPY" in p and "data/raw" in p for p in self.svc.readiness(self.NOW)["problems_ar"]))
        weekend = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
        self.assertTrue(any("السوق مغلق" in p for p in self.svc.readiness(weekend)["problems_ar"]))

    def test_symbol_recommendation_accepts_the_ui_slash_form(self) -> None:
        self.decide("B1/XAUUSD", "SELL", 0.1)
        self.assertEqual(normalize_symbol("xau/usd"), "XAUUSD")
        self.assertEqual(self.svc.recommendation("XAU/USD", self.NOW)["model"], "B1/XAUUSD")
        self.assertEqual([m["model"] for m in self.svc.status("XAU/USD")["models"]], ["B1/XAUUSD"])
        self.assertIsNone(self.svc.recommendation("EURUSD", self.NOW))

    def test_both_variants_signalling_prefers_the_higher_expected_r(self) -> None:
        self.decide("A/EURJPY", "BUY", 0.15)
        self.decide("B1/EURJPY", "BUY", 0.25)
        self.assertEqual(self.svc.recommendation("EURJPY", self.NOW)["model"], "B1/EURJPY")

    def test_nothing_active(self) -> None:
        self.assertEqual(self.svc.best_recommendation(self.NOW), (None, []))
        self.assertEqual(self.svc.closest_symbol(), "EURJPY")
        self.svc.state.decisions["B1/XAUUSD"] = LatestDecision(self.NOW.isoformat(), "NONE", -0.4)
        self.assertEqual(self.svc.closest_symbol(), "XAUUSD")

    def test_quote_to_usd_uses_the_stored_closes(self) -> None:
        store = self.svc.store
        store.append("EURJPY", [(600, 160.0, 160.0, 160.0, 160.0)])
        store.append("EURUSD", [(600, 1.1, 1.1, 1.1, 1.1)])
        self.assertAlmostEqual(self.svc.quote_to_usd("AUDJPY"), 1.1 / 160.0)
        self.assertEqual(self.svc.quote_to_usd("GBP/USD"), 1.0)
        self.assertIsNone(self.svc.quote_to_usd("EURGBP"))


class LifespanTests(unittest.TestCase):
    def test_background_worker_follows_the_app_lifespan(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.web.app import _run_with_app

        events = []

        class Worker:
            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

        app = FastAPI()
        _run_with_app(app, Worker())
        with TestClient(app):
            self.assertEqual(events, ["start"])
        self.assertEqual(events, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()

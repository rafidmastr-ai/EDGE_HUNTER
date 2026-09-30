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
from app.ml_edge.live import SYMBOLS, EdgeMLService, market_open
from app.ml_edge.live_store import M1Store, StoredSeries, fetch_updates
from app.ml_edge.model import SymbolModel
from app.ml_edge.walkforward import execution_config
from app.providers.models import LiveProviderError
from app.research.intraday import Bars, make_orders, simulate

START = 1_767_571_200  # 2026-01-05 00:00 UTC (Monday)
PRICES = {"AUDUSD": 0.66, "EURJPY": 160.0, "EURUSD": 1.08, "GBPUSD": 1.27, "NZDUSD": 0.60, "XAUUSD": 2400.0}


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
        provider = FakeProvider({1: "live_empty_data", 2: "provider_local_rate_limited"})
        added = fetch_updates(store, provider, "EURUSD", last + timedelta(minutes=2000), page_minutes=700, pause=lambda s: None)
        self.assertEqual(added, 0)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[1][0], provider.calls[0][1] + timedelta(minutes=1))

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
        latest = svc.status("EURJPY")["models"][0]["latest"]
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
        svc.refresh(datetime.fromtimestamp(last + 3 * 3600, tz=timezone.utc), fetch=False)
        self.assertFalse(svc.status("EURJPY")["models"][0]["latest"]["fresh"])


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

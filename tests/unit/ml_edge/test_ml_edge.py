from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.ml_edge.data import SymbolData, cost_price
from app.ml_edge.features import build_all
from app.ml_edge.labels import LabelConfig, triple_barrier
from app.ml_edge.model import SymbolModel, edges, make_regressor
from app.ml_edge.walkforward import EMBARGO, Rows, _train_mask
from app.research.intraday import DAY, Bars, slice_time

START = 1_704_067_200  # 2024-01-01 00:00 UTC (Monday)


def random_symbol(name: str, days: int = 40, seed: int = 1, price: float = 1.1) -> SymbolData:
    rng = random.Random(seed)
    times, rows = [], []
    for minute in range(days * 1440):
        t = START + minute * 60
        if (t // 86400 + 3) % 7 >= 5:
            continue
        close = price * (1 + rng.gauss(0, 0.0003))
        rows.append((price, max(price, close) * (1 + abs(rng.gauss(0, 0.0001))), min(price, close) * (1 - abs(rng.gauss(0, 0.0001))), close))
        times.append(t)
        price = close
    arr = np.array(rows)
    n = len(arr)
    return SymbolData(name, Bars(np.array(times, dtype=np.int64), *arr.T, 60), np.ones(n) * 5, np.ones(n) * 10)


def truncate(data: SymbolData, end: int) -> SymbolData:
    m1 = slice_time(data.m1, end=end)
    return SymbolData(data.symbol, m1, data.tick_volume[: len(m1)], data.spread[: len(m1)])


def features_are_causal(build, symbols: dict[str, SymbolData], cut: int) -> bool:
    full = build(symbols)
    part = build({s: truncate(d, cut) for s, d in symbols.items()})
    for s in symbols:
        a, b = full[s], part[s]
        known = a.close_time <= cut
        if not np.array_equal(a.close_time[known], b.close_time[: known.sum()]):
            return False
        if not np.allclose(a.x[known], b.x[: known.sum()], equal_nan=True, rtol=1e-5, atol=1e-6):
            return False
    return True


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols = {"EURUSD": random_symbol("EURUSD", seed=1), "XAUUSD": random_symbol("XAUUSD", seed=2, price=2000.0)}
        self.cut = int(self.symbols["EURUSD"].m1.open_time[len(self.symbols["EURUSD"].m1) * 3 // 4] // 3600 * 3600)

    def test_features_do_not_change_when_future_bars_are_removed(self) -> None:
        self.assertTrue(features_are_causal(build_all, self.symbols, self.cut))

    def test_the_causality_check_catches_a_leaky_feature(self) -> None:
        def leaky(symbols):
            frames = build_all(symbols)
            out = {}
            for s, f in frames.items():
                x = f.x.copy()
                x[:-1, 0] = x[1:, 0]  # next bar's value: look-ahead
                out[s] = type(f)(f.symbol, f.close_time, x, f.names, f.atr, f.tradable)
            return out

        self.assertFalse(features_are_causal(leaky, self.symbols, self.cut))

    def test_same_feature_set_for_every_symbol(self) -> None:
        frames = build_all(self.symbols)
        self.assertEqual(frames["EURUSD"].names, frames["XAUUSD"].names)
        self.assertIn("usd_strength_4", frames["EURUSD"].names)


class LabelTests(unittest.TestCase):
    def m1(self, rows, start=START + 10 * 3600) -> Bars:
        arr = np.array(rows, dtype=float)
        return Bars(start + 60 * np.arange(len(arr)), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], 60)

    def test_barrier_order_stop_first_and_time_exit(self) -> None:
        flat = [(100, 100.1, 99.9, 100)] * 5
        up = self.m1(flat + [(100, 101.5, 99.9, 101)] + flat)
        both = self.m1(flat + [(100, 101.5, 98.5, 100)] + flat)
        none = self.m1(flat + [(100, 100.5, 99.8, 100.4)] + [(100.4, 100.5, 100.3, 100.4)] * 300)
        cfg = LabelConfig(1.0, 8)
        at = np.array([up.open_time[5]])
        atr = np.array([1.0])
        self.assertEqual(triple_barrier(up, at, atr, cfg)[0], 1.0)
        self.assertEqual(triple_barrier(both, at, atr, cfg)[0], -1.0)
        value = triple_barrier(none, at, atr, cfg)[0]
        self.assertAlmostEqual(value, 0.4)  # exit at the close after 8 x 15 M1 bars

    def test_session_cutoff(self) -> None:
        start = START + 20 * 3600 + 30 * 60  # 20:30 UTC
        m1 = self.m1([(100, 100.1, 99.9, 100)] * 14 + [(100, 100.1, 99.9, 100.3)] + [(100.3, 102, 100, 101.9)] * 20, start=start)
        value = triple_barrier(m1, np.array([start]), np.array([1.0]), LabelConfig(1.0, 16))[0]
        self.assertAlmostEqual(value, 0.3)  # closed at the 20:45 open, before the later rally


class HoldLabelTests(unittest.TestCase):
    def test_hold_label_matches_execution_and_counts_swap_nights(self) -> None:
        from app.ml_edge.labels import triple_barrier_exits
        from app.ml_edge.model import expected_nights
        from app.research.intraday import ExecutionConfig, make_orders, simulate

        data = random_symbol("EURUSD", days=12, seed=5)
        frames = build_all({"EURUSD": data}, session_start_hour=1)
        f = frames["EURUSD"].select(frames["EURUSD"].tradable)
        pick = np.arange(len(f.close_time))[::37]
        cfg = LabelConfig(2.0, 48, max_hold_minutes=720)
        y, entry, exit_ = triple_barrier_exits(data.m1, f.close_time[pick], f.atr[pick], cfg)
        ok = np.isfinite(y)
        orders = make_orders(f.close_time[pick][ok], np.ones(ok.sum(), int), np.full(ok.sum(), np.nan), np.full(ok.sum(), np.nan),
                             stop_distance=2.0 * f.atr[pick][ok], target_rr=np.ones(ok.sum()))
        trades = simulate(data.m1, orders, ExecutionConfig(cost=0.0, session_exit=None, max_hold_seconds=43200, weekend_exit=(20, 45),
                                                           one_position=False, stop_fills_at_gap=False, swap_per_night=0.0))
        self.assertEqual(len(trades), int(ok.sum()))
        self.assertTrue(np.allclose(trades.r_gross, y[ok]))
        self.assertTrue(np.array_equal(trades.exit_time, exit_[ok]))
        self.assertTrue(np.all(exit_[ok] - entry[ok] <= 43200))
        tuesday_18 = START + DAY + 18 * 3600  # 2024-01-02 18:00 UTC
        self.assertEqual(expected_nights(np.array([tuesday_18]), cfg)[0], 1)
        self.assertEqual(expected_nights(np.array([tuesday_18]), LabelConfig(2.0, 16))[0], 0)


class WalkForwardTests(unittest.TestCase):
    def test_training_rows_stop_one_day_before_validation(self) -> None:
        t = START + 900 * np.arange(2000)
        r = Rows("X", t, np.zeros((2000, 1)), np.ones(2000), {"k": np.zeros(2000)})
        boundary = int(t[1500])
        mask = _train_mask(r, "k", boundary)
        self.assertTrue(np.all(t[mask] < boundary - EMBARGO))
        self.assertEqual(EMBARGO, DAY)

    def test_permuted_target_gives_no_skill(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(6000, 5)).astype(np.float32)
        y = np.tanh(x[:, 0] + rng.normal(scale=1.0, size=6000))
        model = make_regressor("small").fit(x[:4000], rng.permutation(y[:4000]))
        pred = model.predict(x[4000:])
        self.assertLess(abs(np.corrcoef(pred, y[4000:])[0, 1]), 0.1)
        real = make_regressor("small").fit(x[:4000], y[:4000]).predict(x[4000:])
        self.assertGreater(np.corrcoef(real, y[4000:])[0, 1], 0.3)


class ModelTests(unittest.TestCase):
    def test_edges_are_symmetric_and_cost_aware(self) -> None:
        edge, direction = edges(np.array([0.3, -0.3, 0.0]), np.array([0.1, 0.1, 0.1]))
        self.assertTrue(np.allclose(edge, [0.2, 0.2, -0.1]))
        self.assertEqual(direction.tolist(), [1, -1, 1])

    def test_save_load_roundtrip_and_checksum(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3000, 4)).astype(np.float32)
        y = np.tanh(x[:, 0])
        g = make_regressor("small").fit(x, y)
        s = make_regressor("small", symbol_level=True).fit(np.column_stack([x, g.predict(x)]), y)
        model = SymbolModel("EURUSD", "stacked", "small", LabelConfig(1.0, 8), ("a", "b", "c", "d"), 0.10, True,
                            cost_price("EURUSD"), g, s, warmup_days=1)
        atr = np.full(len(x), 0.001)
        times = START + 900 * np.arange(len(x))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            loaded = SymbolModel.load(Path(tmp))
            decisions = model.decide(times, x, atr)[0]
            self.assertTrue(decisions.any())
            self.assertTrue(np.array_equal(loaded.decide(times, x, atr)[0], decisions))
            blob = Path(tmp) / "model.joblib"
            blob.write_bytes(blob.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                SymbolModel.load(Path(tmp))

    def test_disabled_model_never_trades(self) -> None:
        x = np.zeros((10, 1), dtype=np.float32)
        g = make_regressor("small").fit(np.r_[x, x + 1], np.r_[np.zeros(10), np.ones(10)])
        model = SymbolModel("XAUUSD", "global", "small", LabelConfig(1.0, 8), ("a",), 0.5, False, 0.1, g, None)
        self.assertFalse(model.decide(START + 900 * np.arange(10), x + 1, np.ones(10))[0].any())

    def test_rolling_threshold_uses_only_previous_days(self) -> None:
        from app.ml_edge.model import rolling_thresholds

        times = START + 900 * np.arange(96 * 30)  # 30 days, 96 rows per day
        edge = np.arange(len(times), dtype=float)
        thr = rolling_thresholds(times, edge, 0.10, window_days=5, warmup_days=3, min_history=50)
        self.assertTrue(np.isinf(thr[: 96 * 3]).all())
        day10 = 96 * 10
        expected = np.quantile(edge[96 * 5 : day10], 0.90)
        self.assertAlmostEqual(thr[day10], expected)
        edge_changed = edge.copy()
        edge_changed[day10:] = -1e9  # later edges must not affect day-10 thresholds
        self.assertAlmostEqual(rolling_thresholds(times, edge_changed, 0.10, window_days=5, warmup_days=3, min_history=50)[day10], expected)


if __name__ == "__main__":
    unittest.main()

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
        model = SymbolModel("EURUSD", "stacked", "small", LabelConfig(1.0, 8), ("a", "b", "c", "d"), 0.05, True,
                            cost_price("EURUSD"), g, s)
        atr = np.full(len(x), 0.001)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            loaded = SymbolModel.load(Path(tmp))
            self.assertTrue(np.array_equal(loaded.decide(x, atr)[0], model.decide(x, atr)[0]))
            blob = Path(tmp) / "model.joblib"
            blob.write_bytes(blob.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                SymbolModel.load(Path(tmp))

    def test_disabled_model_never_trades(self) -> None:
        x = np.zeros((10, 1), dtype=np.float32)
        g = make_regressor("small").fit(np.r_[x, x + 1], np.r_[np.zeros(10), np.ones(10)])
        model = SymbolModel("XAUUSD", "global", "small", LabelConfig(1.0, 8), ("a",), -1.0, False, 0.1, g, None)
        self.assertFalse(model.decide(x + 1, np.ones(10))[0].any())


if __name__ == "__main__":
    unittest.main()

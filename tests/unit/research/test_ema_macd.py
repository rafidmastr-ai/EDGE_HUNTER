from __future__ import annotations

import random
import unittest

import numpy as np

from app.research.ema_macd import (
    confirmed_swings,
    ema_macd_orders,
    ema_macd_signals,
    flat_threshold,
    hidden_divergence,
    macd,
)
from app.research.intraday import Bars, ExecutionConfig, make_orders, simulate, slice_time
from app.research.intraday_strategies import Context

START = 1_719_792_000  # 2024-07-01 00:00 UTC (Monday)


def bars_from(lows_highs: list[tuple[float, float]], period: int = 300) -> Bars:
    arr = np.array(lows_highs, dtype=float)
    mid = arr.mean(axis=1)
    return Bars(START + period * np.arange(len(arr)), mid, arr[:, 1], arr[:, 0], mid, period)


def random_m1(days: int = 25, seed: int = 8) -> Bars:
    rng = random.Random(seed)
    times, rows, price = [], [], 2300.0
    for minute in range(days * 1440):
        t = START + minute * 60
        if (t // 86400 + 3) % 7 >= 5:
            continue
        drift = 0.02 * np.sin(minute / 3000.0)
        close = price + drift + rng.gauss(0, 0.3)
        rows.append((price, max(price, close) + abs(rng.gauss(0, 0.15)), min(price, close) - abs(rng.gauss(0, 0.15)), close))
        times.append(t)
        price = close
    arr = np.array(rows)
    return Bars(np.array(times, dtype=np.int64), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], 60)


class IndicatorTests(unittest.TestCase):
    def test_macd_matches_hand_computed_recursion(self) -> None:
        close = np.linspace(100, 140, 60) + np.sin(np.arange(60))
        line, sig = macd(close)

        def ema_ref(values, span):
            alpha, out = 2 / (span + 1), [values[0]]
            for value in values[1:]:
                out.append(out[-1] + alpha * (value - out[-1]))
            return np.array(out)

        ref_line = ema_ref(close, 12) - ema_ref(close, 26)
        ref_sig = ema_ref(ref_line, 9)
        self.assertTrue(np.isnan(line[:35]).all())
        self.assertTrue(np.allclose(line[35:], ref_line[35:]))
        self.assertTrue(np.allclose(sig[35:], ref_sig[35:]))

    def test_swing_is_invisible_until_k_bars_after_it(self) -> None:
        # swing low at index 5 (price 90); k = 3 -> usable from index 8.
        rows = [(100 - i, 110) for i in range(5)] + [(90, 110)] + [(95 + i, 110) for i in range(6)]
        swings = confirmed_swings(bars_from(rows), 3)
        self.assertTrue(np.isnan(swings["low"][:8]).all())
        self.assertEqual(swings["low"][8], 90)
        self.assertEqual(swings["low_bar"][8], 5)

    def test_hidden_divergence(self) -> None:
        n = 30
        swings = {name: np.full(n, np.nan) for name in ("low", "prev_low", "high", "prev_high")}
        swings.update({name: np.full(n, -1) for name in ("low_bar", "prev_low_bar", "high_bar", "prev_high_bar")})
        swings["prev_low"][20], swings["low"][20] = 100.0, 105.0  # higher low in price
        swings["prev_low_bar"][20], swings["low_bar"][20] = 5, 15
        line = np.zeros(n)
        line[5], line[15] = -1.0, -2.0  # lower low in MACD
        bull, bear = hidden_divergence(swings, line)
        self.assertTrue(bull[20])
        self.assertFalse(bear.any())
        line[15] = -0.5  # regular behaviour -> no hidden divergence
        self.assertFalse(hidden_divergence(swings, line)[0][20])


class SignalTests(unittest.TestCase):
    def test_cross_must_be_below_zero_above_ema_and_not_flat(self) -> None:
        m1 = random_m1()
        bars = Context(m1).bars(5)
        long, short = ema_macd_signals(bars, threshold=0.0)
        line, sig = macd(bars.close)
        from app.research.intraday import ema

        e200 = ema(bars.close, 200)
        for t in np.flatnonzero(long):
            self.assertTrue(line[t - 1] <= sig[t - 1] and line[t] > sig[t] and line[t] < 0 and bars.close[t] > e200[t])
        for t in np.flatnonzero(short):
            self.assertTrue(line[t - 1] >= sig[t - 1] and line[t] < sig[t] and line[t] > 0 and bars.close[t] < e200[t])
        strict_long, _ = ema_macd_signals(bars, threshold=1.0)  # every bar counts as flat
        self.assertFalse(strict_long.any())

    def test_orders_do_not_change_when_future_bars_are_removed(self) -> None:
        full = random_m1()
        cut = int(full.open_time[len(full) * 3 // 4] // 3600 * 3600)
        threshold = flat_threshold(Context(full).bars(5), train_end=cut, percentile=20)
        for variant in ("core", "atr", "hidden_div"):
            with self.subTest(variant=variant):
                a, _ = ema_macd_orders(Context(full), 5, variant=variant, threshold=threshold)
                b, _ = ema_macd_orders(Context(slice_time(full, end=cut)), 5, variant=variant, threshold=threshold)
                known = a.signal_time <= cut
                self.assertGreater(known.sum(), 0)
                self.assertEqual(a.signal_time[known].tolist(), b.signal_time.tolist())
                self.assertTrue(np.allclose(a.stop[known], b.stop, equal_nan=True))

    def test_core_stop_is_below_last_confirmed_swing_and_target_from_fill(self) -> None:
        m1 = random_m1()
        ctx = Context(m1)
        bars = ctx.bars(15)
        threshold = flat_threshold(bars, train_end=int(m1.open_time[-1]), percentile=20)
        orders, _ = ema_macd_orders(ctx, 15, variant="core", threshold=threshold)
        swings = confirmed_swings(bars, 3)
        index = np.searchsorted(bars.close_time, orders.signal_time)
        for i, t in enumerate(index):
            expected = swings["low"][t] - 0.01 if orders.direction[i] > 0 else swings["high"][t] + 0.01
            self.assertAlmostEqual(orders.stop[i], expected)
            self.assertLessEqual(swings["low_bar" if orders.direction[i] > 0 else "high_bar"][t] + 3, t)
        trades = simulate(m1, orders, ExecutionConfig(cost=0.0, session_exit=None, last_entry=None))
        self.assertGreater(len(trades), 0)
        for entry, stop, exit_, outcome, d in zip(trades.entry, trades.stop, trades.exit, trades.outcome, trades.direction):
            if outcome == 1:
                self.assertAlmostEqual(exit_, entry + d * 1.5 * abs(entry - stop))
        self.assertTrue(np.all(trades.entry_time >= trades.signal_time))


class ExecutionExtensionTests(unittest.TestCase):
    def test_stop_distance_and_rr_are_applied_at_fill(self) -> None:
        rows = [(100.0, 100.5, 99.5, 100.0)] * 3 + [(101.0, 104.5, 100.8, 104.0)] + [(104.0, 104.1, 103.9, 104.0)] * 3
        arr = np.array(rows)
        m1 = Bars(START + 36000 + 60 * np.arange(len(arr)), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], 60)
        orders = make_orders([m1.open_time[3]], [1], [np.nan], [np.nan], stop_distance=np.array([2.0]), target_rr=np.array([1.5]))
        stats: dict = {}
        trades = simulate(m1, orders, ExecutionConfig(cost=0.0), stats)
        self.assertEqual((trades.entry[0], trades.stop[0], trades.exit[0], trades.outcome[0]), (101.0, 99.0, 104.0, 1))
        self.assertEqual(sum(stats.values()), 0)


if __name__ == "__main__":
    unittest.main()

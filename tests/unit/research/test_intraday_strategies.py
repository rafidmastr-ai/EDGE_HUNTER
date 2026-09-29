from __future__ import annotations

import random
import unittest
from datetime import datetime, timezone

import numpy as np

from app.research.intraday import Bars, slice_time
from app.research.intraday_strategies import GENERATORS, Context, ny_orb

START = int(datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp())  # Monday, US summer time


def random_m1(days: int = 12, seed: int = 4) -> Bars:
    rng = random.Random(seed)
    times, rows, price = [], [], 2300.0
    for minute in range(days * 1440):
        t = START + minute * 60
        weekday = (t // 86400 + 3) % 7  # 0 = Monday
        if weekday >= 5:
            continue
        move = rng.gauss(0, 0.35)
        close = price + move
        rows.append((price, max(price, close) + abs(rng.gauss(0, 0.2)), min(price, close) - abs(rng.gauss(0, 0.2)), close))
        times.append(t)
        price = close
    arr = np.array(rows)
    return Bars(np.array(times, dtype=np.int64), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], 60)


class NoLookAheadTests(unittest.TestCase):
    def test_orders_do_not_change_when_future_bars_are_removed(self) -> None:
        full = random_m1()
        cut = int(full.open_time[len(full) * 2 // 3] // 3600 * 3600)
        truncated = slice_time(full, end=cut)
        for name, (generator, space) in GENERATORS.items():
            params = {key: values[0] for key, values in space.items()}
            with self.subTest(strategy=name):
                a = generator(Context(full), **params)
                b = generator(Context(truncated), **params)
                known = a.signal_time <= cut
                self.assertEqual(a.signal_time[known].tolist(), b.signal_time.tolist())
                self.assertTrue(np.allclose(a.stop[known], b.stop))
                self.assertTrue(np.all(a.signal_time % 300 == 0))

    def test_ny_orb_uses_new_york_open_in_utc(self) -> None:
        orders = ny_orb(Context(random_m1()), range_minutes=30, stop_mode="opposite", rr=2.0, bias="none")
        self.assertGreater(len(orders), 0)
        seconds = orders.signal_time % 86400
        # 09:30 New York = 13:30 UTC in July; first break comes after the 30-minute range.
        self.assertTrue(np.all(seconds >= 14 * 3600 + 5 * 60))
        self.assertEqual(len(set((orders.signal_time // 86400).tolist())), len(orders))


if __name__ == "__main__":
    unittest.main()

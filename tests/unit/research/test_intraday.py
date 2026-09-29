from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

from app.research.intraday import (
    Bars,
    ExecutionConfig,
    align_to,
    daily_levels,
    leave_one_year_out,
    make_orders,
    resample,
    simulate,
)

DAY0 = int(datetime(2024, 3, 5, tzinfo=timezone.utc).timestamp())  # a Tuesday


def m1(prices: list[tuple[float, float, float, float]], start: int = DAY0 + 10 * 3600) -> Bars:
    arr = np.array(prices, dtype=float)
    return Bars(start + 60 * np.arange(len(arr)), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], 60)


def flat(n: int, price: float = 100.0) -> list[tuple[float, float, float, float]]:
    return [(price, price + 0.1, price - 0.1, price)] * n


class ExecutionTests(unittest.TestCase):
    def test_market_order_fills_next_open_and_hits_target(self) -> None:
        bars = m1(flat(3) + [(101, 103, 100.5, 102.5)] + flat(5, 102.5))
        orders = make_orders([bars.open_time[3]], [1], [99.0], [102.0])
        trades = simulate(bars, orders, ExecutionConfig(cost=0.0))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.entry[0], 101.0)
        self.assertEqual(trades.outcome[0], 1)
        self.assertAlmostEqual(trades.r_net[0], 0.5)

    def test_signal_never_fills_before_it_is_known(self) -> None:
        bars = m1([(100, 110, 90, 100)] + flat(5))
        orders = make_orders([bars.open_time[0] + 30], [1], [95.0], [105.0])  # mid-bar decision
        trades = simulate(bars, orders, ExecutionConfig(cost=0.0))
        self.assertEqual(trades.entry_time[0], bars.open_time[1])
        self.assertEqual(trades.outcome[0], 0)

    def test_stop_first_when_both_levels_in_one_bar(self) -> None:
        bars = m1(flat(2) + [(100, 103, 98, 100)] + flat(3))
        orders = make_orders([bars.open_time[1]], [1], [99.0], [102.0])
        trades = simulate(bars, orders, ExecutionConfig(cost=0.0))
        self.assertEqual(trades.outcome[0], -1)
        self.assertAlmostEqual(trades.r_net[0], -1.0)

    def test_cost_is_cost_over_risk(self) -> None:
        bars = m1(flat(2) + [(100, 100.1, 97, 98)] + flat(3, 98))
        orders = make_orders([bars.open_time[1]], [1], [98.0], [110.0])
        trades = simulate(bars, orders, ExecutionConfig(cost=0.5))
        self.assertAlmostEqual(trades.r_net[0], -1.0 - 0.25)

    def test_limit_fills_only_when_traded_through_and_expires(self) -> None:
        bars = m1(flat(10) + [(100, 100.2, 99.4, 99.6)] + flat(10, 99.6))
        hit = make_orders([bars.open_time[1]], [1], [98.0], [103.0], limit=np.array([99.5]), expiry_seconds=3600)
        miss = make_orders([bars.open_time[1]], [1], [98.0], [103.0], limit=np.array([99.0]), expiry_seconds=3600)
        expired = make_orders([bars.open_time[1]], [1], [98.0], [103.0], limit=np.array([99.5]), expiry_seconds=300)
        trades = simulate(bars, hit, ExecutionConfig(cost=0.0))
        self.assertEqual((trades.entry[0], trades.entry_time[0]), (99.5, bars.open_time[10]))
        self.assertEqual(len(simulate(bars, miss, ExecutionConfig(cost=0.0))), 0)
        self.assertEqual(len(simulate(bars, expired, ExecutionConfig(cost=0.0))), 0)

    def test_session_exit_and_no_late_entries(self) -> None:
        start = DAY0 + 20 * 3600 + 30 * 60  # 20:30 UTC
        bars = m1(flat(40), start=start)
        late = make_orders([start + 60 * 31], [1], [99.0], [105.0])  # 21:01, after last entry
        held = make_orders([DAY0 + 20 * 3600], [1], [99.0], [105.0])  # 20:00 decision
        self.assertEqual(len(simulate(bars, late, ExecutionConfig(cost=0.0))), 0)
        trades = simulate(bars, held, ExecutionConfig(cost=0.0))
        self.assertEqual(trades.exit_time[0], DAY0 + 20 * 3600 + 45 * 60)
        self.assertEqual(trades.outcome[0], 0)

    def test_one_position_at_a_time(self) -> None:
        bars = m1(flat(30))
        orders = make_orders(bars.open_time[[1, 2, 3]], [1, 1, 1], [99.0] * 3, [105.0] * 3)
        self.assertEqual(len(simulate(bars, orders, ExecutionConfig(cost=0.0))), 1)
        self.assertEqual(len(simulate(bars, orders, ExecutionConfig(cost=0.0, one_position=False))), 3)

    def test_breakeven_and_trailing_stop(self) -> None:
        path = flat(2) + [(100, 102.2, 99.9, 102)] + [(102, 102.1, 99.95, 100)] + flat(3, 100)
        bars = m1(path)
        be = make_orders([bars.open_time[1]], [1], [99.0], [np.nan], breakeven_r=1.0)
        trades = simulate(bars, be, ExecutionConfig(cost=0.0))
        self.assertEqual((trades.outcome[0], trades.r_net[0]), (2, 0.0))
        trail = make_orders([bars.open_time[1]], [1], [99.0], [np.nan], trail_distance=np.array([1.0]))
        trades = simulate(bars, trail, ExecutionConfig(cost=0.0))
        self.assertAlmostEqual(trades.exit[0], 101.2)


class DataToolsTests(unittest.TestCase):
    def test_resample_and_alignment_are_causal(self) -> None:
        bars = m1([(i, i + 0.5, i - 0.5, i + 0.2) for i in range(120)], start=DAY0 + 10 * 3600)
        h1 = resample(bars, 60)
        self.assertEqual(len(h1), 2)
        self.assertEqual((h1.open[0], h1.close[0], h1.high[0]), (0, 59.2, 59.5))
        m5 = resample(bars, 5)
        aligned = align_to(m5, h1, h1.close)
        self.assertTrue(np.isnan(aligned[:11]).all())  # first H1 not closed yet
        self.assertEqual(aligned[11], 59.2)

    def test_asia_levels_only_after_asia_closes(self) -> None:
        start = DAY0
        bars = m1([(100 + (i < 420) * 5, 106 if i == 100 else 101, 99, 100) for i in range(600)], start=start)
        m15 = resample(bars, 15)
        levels = daily_levels(bars, m15)
        close_hours = (m15.close_time - start) / 3600
        self.assertTrue(np.isnan(levels["asia_high"][close_hours < 7]).all())
        self.assertEqual(levels["asia_high"][close_hours >= 7][0], 106)

    def test_leave_one_year_out_never_uses_the_held_out_year(self) -> None:
        from app.research.intraday import Trades

        def trades(values_by_year: dict[int, float]) -> Trades:
            times, r = [], []
            for year, value in values_by_year.items():
                base = int(datetime(year, 6, 1, tzinfo=timezone.utc).timestamp())
                times += [base + i * 3600 for i in range(40)]
                r += [value] * 40
            n = len(times)
            z = np.zeros(n)
            t = np.array(times)
            return Trades(t, t, t, np.ones(n, int), z, z, z, np.array(r), np.array(r), np.zeros(n, int))

        results = {"a": trades({2020: 1.0, 2021: 1.0, 2022: -5.0}), "b": trades({2020: 0.1, 2021: 0.1, 2022: 0.1})}
        out = leave_one_year_out(results, [2020, 2021, 2022])
        self.assertEqual(out["picked"][2022], "a")  # chosen on 2020-2021 only
        self.assertEqual(out["held_out_avg_r_by_year"][2022], -5.0)
        self.assertEqual(out["picked"][2020], "b")


if __name__ == "__main__":
    unittest.main()

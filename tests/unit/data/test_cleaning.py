from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from app.data.cleaning import drop_synthetic_flat_runs
from app.data.pipeline import HistoricalDataPipeline
from app.data.schema import CanonicalOHLC

T0 = datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc)  # Friday


def bar(ts: datetime, o: str, h: str, l: str, c: str) -> CanonicalOHLC:
    return CanonicalOHLC(timestamp=ts, open=Decimal(o), high=Decimal(h), low=Decimal(l), close=Decimal(c), volume=None)


def moving(n: int, start: datetime) -> list[CanonicalOHLC]:
    return [bar(start + timedelta(minutes=i), "1.1000", "1.1004", "1.0998", "1.1002") for i in range(n)]


def flat(n: int, start: datetime, price: str = "1.1002") -> list[CanonicalOHLC]:
    return [bar(start + timedelta(minutes=i), price, price, price, price) for i in range(n)]


class DropSyntheticFlatRunsTests(unittest.TestCase):
    def test_long_flat_run_repeating_last_close_is_removed(self) -> None:
        before = moving(60, T0)
        filler = flat(120, T0 + timedelta(minutes=60))  # e.g. Sunday filler
        after = moving(60, T0 + timedelta(minutes=180))
        cleaned, removed = drop_synthetic_flat_runs(before + filler + after)
        self.assertEqual(removed, 120)
        self.assertEqual(cleaned, before + after)

    def test_run_after_a_gap_is_removed(self) -> None:
        before = moving(60, T0)
        filler = flat(45, T0 + timedelta(days=2))  # starts after a weekend gap
        after = moving(30, T0 + timedelta(days=2, minutes=45))
        cleaned, removed = drop_synthetic_flat_runs(before + filler + after)
        self.assertEqual(removed, 45)

    def test_short_quiet_periods_are_kept(self) -> None:
        series = moving(60, T0) + flat(10, T0 + timedelta(minutes=60)) + moving(60, T0 + timedelta(minutes=70))
        cleaned, removed = drop_synthetic_flat_runs(series)
        self.assertEqual(removed, 0)
        self.assertEqual(len(cleaned), len(series))

    def test_flat_bars_at_changing_price_are_kept(self) -> None:
        series = [bar(T0 + timedelta(minutes=i), p, p, p, p) for i, p in enumerate(f"1.{1000 + i}" for i in range(40))]
        self.assertEqual(drop_synthetic_flat_runs(series)[1], 0)

    def test_higher_native_timeframe_needs_three_bars(self) -> None:
        hourly = [bar(T0 + timedelta(hours=i), "1.1", "1.2", "1.0", "1.1") for i in range(10)]
        single = hourly + [bar(T0 + timedelta(hours=10), "1.1", "1.1", "1.1", "1.1")]
        self.assertEqual(drop_synthetic_flat_runs(single)[1], 0)
        triple = hourly + [bar(T0 + timedelta(hours=10 + i), "1.1", "1.1", "1.1", "1.1") for i in range(3)]
        self.assertEqual(drop_synthetic_flat_runs(triple)[1], 3)

    def test_pipeline_reports_removed_rows(self) -> None:
        series = moving(40, T0) + flat(60, T0 + timedelta(minutes=40)) + moving(40, T0 + timedelta(minutes=100))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EURUSD.csv"
            path.write_text("timestamp,open,high,low,close\n" + "".join(
                f"{b.timestamp.strftime('%Y-%m-%d %H:%M:%S')},{b.open},{b.high},{b.low},{b.close}\n" for b in series), encoding="utf-8")
            bars, report = HistoricalDataPipeline().load(path, "EURUSD", "M1")
        self.assertEqual(len(bars), 80)
        self.assertEqual(report.synthetic_flat_bars_removed, 60)
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()

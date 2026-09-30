from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.data.mt5_history import is_mt5_symbol_dir, load_mt5_symbol, server_to_utc

ATHENS = ZoneInfo("Europe/Athens")
HEADER = "timestamp,open,high,low,close,tick_volume,spread\n"


def row(stamp: str, price: float) -> str:
    return f"{stamp}+00:00,{price},{price + 0.5},{price - 0.5},{price + 0.1},10,15\n"


class ServerToUtcTests(unittest.TestCase):
    def test_summer_and_winter_offsets(self) -> None:
        # NFP 08:30 New York: 15:30 server time in both cases, 12:30 / 13:30 UTC.
        summer = server_to_utc(datetime(2023, 7, 7, 15, 30, tzinfo=timezone.utc), ATHENS)
        winter = server_to_utc(datetime(2023, 1, 6, 15, 30, tzinfo=timezone.utc), ATHENS)
        self.assertEqual(summer, datetime(2023, 7, 7, 12, 30, tzinfo=timezone.utc))
        self.assertEqual(winter, datetime(2023, 1, 6, 13, 30, tzinfo=timezone.utc))

    def test_eu_not_us_dst_dates(self) -> None:
        # 2023-03-17: US already on DST, EU not yet -> still UTC+2.
        self.assertEqual(
            server_to_utc(datetime(2023, 3, 17, 14, 30), ATHENS), datetime(2023, 3, 17, 12, 30, tzinfo=timezone.utc)
        )

    def test_ambiguous_and_missing_local_times_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            server_to_utc(datetime(2023, 10, 29, 3, 30), ATHENS)  # clock repeats 03:00-04:00
        with self.assertRaises(ValueError):
            server_to_utc(datetime(2023, 3, 26, 3, 30), ATHENS)  # clock skips 03:00-04:00


class LoadMt5SymbolTests(unittest.TestCase):
    def test_merges_years_converts_to_utc_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "XAUUSD"
            folder.mkdir()
            (folder / "SOURCE.json").write_text(json.dumps({"symbol": "XAUUSD", "server_tz": "Europe/Athens"}), encoding="utf-8")
            (folder / "XAUUSD_M1_2023.csv").write_text(
                HEADER + row("2023-12-29T23:58:00", 2000) + row("2023-12-29T23:59:00", 2001), encoding="utf-8"
            )
            (folder / "XAUUSD_M1_2024.csv").write_text(
                HEADER + row("2023-12-29T23:59:00", 2001) + row("2024-07-01T10:00:00", 2300), encoding="utf-8"
            )
            self.assertTrue(is_mt5_symbol_dir(folder))
            symbol, bars, summary = load_mt5_symbol(folder)
        self.assertEqual(symbol, "XAUUSD")
        self.assertEqual(
            [bar.timestamp for bar in bars],
            [
                datetime(2023, 12, 29, 21, 58, tzinfo=timezone.utc),
                datetime(2023, 12, 29, 21, 59, tzinfo=timezone.utc),
                datetime(2024, 7, 1, 7, 0, tzinfo=timezone.utc),
            ],
        )
        self.assertEqual(summary["server_tz"], "Europe/Athens")
        self.assertEqual(len(summary["files"]), 2)

    def test_missing_server_tz_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "EURUSD"
            folder.mkdir()
            (folder / "SOURCE.json").write_text("{}", encoding="utf-8")
            (folder / "EURUSD_M1_2024.csv").write_text(HEADER + row("2024-07-01T10:00:00", 1.1), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mt5_symbol(folder)


if __name__ == "__main__":
    unittest.main()


class RepositoryHistoryTests(unittest.TestCase):
    """The committed MT5 exports all share one server clock (Europe/Athens)."""

    RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

    def test_every_symbol_folder_declares_the_athens_server_clock(self) -> None:
        folders = sorted(p for p in self.RAW.iterdir() if (p / "SOURCE.json").exists())
        self.assertIn("AUDJPY", [p.name for p in folders])
        for folder in folders:
            source = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
            self.assertEqual(source["server_tz"], "Europe/Athens", folder.name)

    def test_audjpy_first_and_last_bars_convert_to_utc(self) -> None:
        from app.ml_edge.data import load_symbol

        data = load_symbol(self.RAW / "AUDJPY")
        first = datetime.fromtimestamp(int(data.m1.open_time[0]), tz=timezone.utc)
        last = datetime.fromtimestamp(int(data.m1.open_time[-1]), tz=timezone.utc)
        self.assertEqual(first, datetime(2026, 1, 1, 22, 1, tzinfo=timezone.utc))  # 00:01 server, UTC+2 in winter
        self.assertEqual(last, datetime(2026, 9, 28, 20, 59, tzinfo=timezone.utc))  # 23:59 server, UTC+3 in summer
        self.assertEqual(len(data.m1), 275853)

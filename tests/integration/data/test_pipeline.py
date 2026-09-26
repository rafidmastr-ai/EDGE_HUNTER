import tempfile
import unittest
from pathlib import Path

from app.data.pipeline import HistoricalDataPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_sorts_and_deduplicates_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "XAUUSD.csv"
            path.write_text(
                "timestamp,open,high,low,close\n"
                "2026-01-01T00:01:00+00:00,101,102,100,101.5\n"
                "2026-01-01T00:00:00+00:00,100,101,99,100.5\n"
                "2026-01-01T00:00:00+00:00,100,101,99,100.5\n",
                encoding="utf-8",
            )
            bars, report = HistoricalDataPipeline().load(path, "XAUUSD", "M1")
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars[0].timestamp.isoformat(), "2026-01-01T00:00:00+00:00")
            self.assertEqual(bars[1].timestamp.isoformat(), "2026-01-01T00:01:00+00:00")
            self.assertTrue(report.is_valid)

    def test_pipeline_rejects_impossible_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text(
                "timestamp,open,high,low,close\n"
                "2026-01-01T00:00:00+00:00,100,101,99,103\n",
                encoding="utf-8",
            )
            bars, report = HistoricalDataPipeline().load(path, "XAUUSD", "M1")
            self.assertEqual(len(bars), 1)
            self.assertEqual(report.invalid_ohlc_rows, 1)
            self.assertFalse(report.is_valid)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from app.data.loaders.csv_loader import CSVLoader


class CSVLoaderTests(unittest.TestCase):
    def test_discover_by_symbol_is_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "XAUUSD_2026.csv").write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            (root / "XAUUSD_2025.csv").write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            (root / "GBPUSD.csv").write_text("timestamp,open,high,low,close\n", encoding="utf-8")

            paths = CSVLoader().discover(root, "XAUUSD")
            self.assertEqual([p.name for p in paths], ["XAUUSD_2025.csv", "XAUUSD_2026.csv"])

    def test_read_rows_preserves_csv_values_as_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text(
                "timestamp,open,high,low,close\n"
                "2026-01-01T00:00:00+00:00,100,101,99,100.5\n",
                encoding="utf-8",
            )
            rows = list(CSVLoader().read_rows(path))
            self.assertEqual(rows[0]["open"], "100")
            self.assertEqual(rows[0]["close"], "100.5")


if __name__ == "__main__":
    unittest.main()

import unittest
from decimal import Decimal

from app.data.normalizers.ohlc_normalizer import normalize_row


class NormalizerTests(unittest.TestCase):
    def test_aliases_and_timezone_are_normalized_to_utc(self):
        bar = normalize_row(
            {
                "time": "2026-01-01T10:00:00+03:00",
                "o": "100",
                "h": "105",
                "l": "99",
                "c": "103",
                "volume": "12",
            }
        )
        self.assertEqual(bar.timestamp.isoformat(), "2026-01-01T07:00:00+00:00")
        self.assertEqual(bar.open, Decimal("100"))
        self.assertEqual(bar.high, Decimal("105"))
        self.assertEqual(bar.low, Decimal("99"))
        self.assertEqual(bar.close, Decimal("103"))
        self.assertEqual(bar.volume, Decimal("12"))

    def test_naive_timestamp_is_treated_as_utc(self):
        bar = normalize_row(
            {
                "timestamp": "2026-01-01T00:00:00",
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
            }
        )
        self.assertEqual(bar.timestamp.isoformat(), "2026-01-01T00:00:00+00:00")

    def test_missing_required_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing close value"):
            normalize_row(
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "open": "1",
                    "high": "2",
                    "low": "0.5",
                }
            )

    def test_invalid_numeric_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid numeric value for open"):
            normalize_row(
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "open": "not-a-number",
                    "high": "2",
                    "low": "0.5",
                    "close": "1.5",
                }
            )

    def test_invalid_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            normalize_row(
                {
                    "timestamp": "not-a-date",
                    "open": "1",
                    "high": "2",
                    "low": "0.5",
                    "close": "1.5",
                }
            )

    def test_epoch_milliseconds_and_seconds_are_supported(self):
        row = {"open": "1", "high": "2", "low": "0.5", "close": "1.5"}
        millis = normalize_row({**row, "timestamp": "1756684800000"})
        seconds = normalize_row({**row, "timestamp": "1756684800"})
        self.assertEqual(millis.timestamp.isoformat(), "2025-09-01T00:00:00+00:00")
        self.assertEqual(seconds.timestamp, millis.timestamp)
        with self.assertRaisesRegex(ValueError, "Invalid timestamp"):
            normalize_row({**row, "timestamp": "12345"})


if __name__ == "__main__":
    unittest.main()

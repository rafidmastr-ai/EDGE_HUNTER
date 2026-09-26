import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.schema import CanonicalOHLC
from app.data.validators.ohlc_validator import OHLCValidator


class ValidatorTests(unittest.TestCase):
    def bar(self, minute: int, close: str = "103") -> CanonicalOHLC:
        return CanonicalOHLC(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal(close),
        )

    def test_valid_sequence(self):
        report = OHLCValidator().validate([self.bar(0), self.bar(1)], "XAUUSD", "M1")
        self.assertEqual(report.total_rows, 2)
        self.assertEqual(report.valid_rows, 2)
        self.assertTrue(report.is_valid)

    def test_duplicate_timestamp_is_reported(self):
        report = OHLCValidator().validate([self.bar(0), self.bar(0)], "XAUUSD", "M1")
        self.assertEqual(report.duplicate_timestamps, 1)
        self.assertFalse(report.is_valid)

    def test_non_monotonic_timestamp_is_reported(self):
        report = OHLCValidator().validate([self.bar(1), self.bar(0)], "XAUUSD", "M1")
        self.assertEqual(report.non_monotonic_rows, 1)
        self.assertFalse(report.is_valid)

    def test_non_positive_price_is_rejected(self):
        bad = self.bar(0)
        bad = CanonicalOHLC(bad.timestamp, Decimal("0"), bad.high, bad.low, bad.close)
        report = OHLCValidator().validate([bad], "XAUUSD", "M1")
        self.assertEqual(report.invalid_price_rows, 1)
        self.assertFalse(report.is_valid)

    def test_impossible_ohlc_relationship_is_rejected(self):
        bad = CanonicalOHLC(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("103"),
        )
        report = OHLCValidator().validate([bad], "XAUUSD", "M1")
        self.assertEqual(report.invalid_ohlc_rows, 1)
        self.assertFalse(report.is_valid)

    def test_negative_volume_is_reported_as_error(self):
        bad = CanonicalOHLC(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=Decimal("-1"),
        )
        report = OHLCValidator().validate([bad], "XAUUSD", "M1")
        self.assertFalse(report.is_valid)
        self.assertTrue(any("invalid volume" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.data.repository import OHLCRepository
from app.data.schema import CanonicalOHLC


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.bars = [
            CanonicalOHLC(base, Decimal("1"), Decimal("2"), Decimal("0.5"), Decimal("1.5")),
            CanonicalOHLC(base + timedelta(minutes=1), Decimal("1.5"), Decimal("2.5"), Decimal("1"), Decimal("2")),
            CanonicalOHLC(base + timedelta(minutes=2), Decimal("2"), Decimal("3"), Decimal("1.5"), Decimal("2.5")),
        ]

    def test_symbol_is_case_insensitive(self):
        repo = OHLCRepository()
        repo.put("xauusd", "M1", self.bars)
        self.assertEqual(repo.get_slice("XAUUSD", "M1"), self.bars)

    def test_date_slice_is_inclusive(self):
        repo = OHLCRepository()
        repo.put("XAUUSD", "M1", self.bars)
        result = repo.get_slice("XAUUSD", "M1", self.bars[0].timestamp, self.bars[1].timestamp)
        self.assertEqual(result, self.bars[:2])

    def test_unknown_symbol_returns_empty_list(self):
        repo = OHLCRepository()
        repo.put("XAUUSD", "M1", self.bars)
        self.assertEqual(repo.get_slice("GBPUSD", "M1"), [])


if __name__ == "__main__":
    unittest.main()

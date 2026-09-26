import unittest
from datetime import datetime, timezone
from decimal import Decimal

from app.domain.market import OHLCBar


class MarketContractTests(unittest.TestCase):
    def test_canonical_ohlc_bar_is_constructible(self):
        bar = OHLCBar(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
        )
        self.assertEqual(bar.close, Decimal("103"))
        self.assertIsNone(bar.volume)


if __name__ == "__main__":
    unittest.main()

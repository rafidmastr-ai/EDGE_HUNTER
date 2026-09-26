from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from app.domain.market import OHLCBar
from app.providers.models import LiveProviderHealth
from app.web.analysis_service import LocalOHLCAnalysisService
from app.web.schemas import AnalyzeRequest


class FakeLiveProvider:
    name = "mock-live"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_ohlc(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> list[OHLCBar]:
        self.calls.append((symbol, timeframe))
        minutes = {"M5": 5, "M15": 15, "H1": 60}[timeframe]
        count = 420
        first = end - timedelta(minutes=minutes * (count - 1))
        bars: list[OHLCBar] = []
        price = Decimal("2400.00")
        for i in range(count):
            ts = first + timedelta(minutes=minutes * i)
            drift = Decimal("0.20") if (i // 25) % 2 == 0 else Decimal("-0.12")
            close = price + drift
            high = max(price, close) + Decimal("0.25")
            low = min(price, close) - Decimal("0.20")
            bars.append(
                OHLCBar(
                    timestamp=ts,
                    open=price,
                    high=high,
                    low=low,
                    close=close,
                    volume=None,
                )
            )
            price = close
        return bars

    def health(self) -> LiveProviderHealth:
        return LiveProviderHealth(provider=self.name, configured=True, status="healthy")


class LiveAnalysisServiceTests(unittest.TestCase):
    def test_live_source_is_mapped_to_canonical_analysis_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeLiveProvider()
            service = LocalOHLCAnalysisService(
                data_root=Path(tmp),
                live_provider=provider,
                data_mode="live",
                live_history_bars=420,
                live_fallback_to_local=False,
            )
            result = service.analyze(AnalyzeRequest(symbol="XAUUSD", risk_percent=1.0, lot_mode="auto"))
            self.assertEqual(result["metadata"]["data_source"], "mock-live")
            self.assertEqual(result["metadata"]["data_mode"], "live")
            self.assertFalse(result["metadata"]["data_fallback_used"])
            self.assertEqual(result["symbol"], "XAUUSD")
            self.assertEqual(result["analyzed_timeframes"], ["M5", "M15", "H1"])
            self.assertEqual([item[1] for item in provider.calls], ["M5", "M15", "H1"])
            self.assertEqual(result["metadata"]["live_provider_health"]["provider"], "mock-live")


if __name__ == "__main__":
    unittest.main()

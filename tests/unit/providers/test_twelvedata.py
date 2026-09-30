from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.error import URLError

from app.providers.models import LiveProviderError
from app.providers.twelvedata import TwelveDataLiveProvider


class FakeTransport:
    def __init__(self, payload: dict, failures: int = 0) -> None:
        self.payload = payload
        self.failures = failures
        self.calls = 0
        self.requests = []

    def __call__(self, request, timeout):
        self.calls += 1
        self.requests.append((request, timeout))
        if self.calls <= self.failures:
            raise URLError("temporary network failure")
        return json.dumps(self.payload).encode("utf-8")


def sample_payload(count: int = 5) -> dict:
    start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    values = []
    price = Decimal("2500.00")
    for i in range(count):
        ts = start + timedelta(minutes=i)
        close = price + Decimal("0.50")
        values.append(
            {
                "datetime": ts.isoformat().replace("+00:00", "Z"),
                "open": str(price),
                "high": str(close + Decimal("0.20")),
                "low": str(price - Decimal("0.10")),
                "close": str(close),
            }
        )
        price = close
    return {"status": "ok", "values": list(reversed(values))}


class TwelveDataProviderTests(unittest.TestCase):
    def test_requires_api_key(self):
        provider = TwelveDataLiveProvider(api_key=None)
        with self.assertRaises(LiveProviderError) as raised:
            provider.get_ohlc(
                "XAUUSD",
                "M5",
                datetime(2026, 9, 24, tzinfo=timezone.utc),
                datetime(2026, 9, 25, tzinfo=timezone.utc),
            )
        self.assertEqual(raised.exception.code, "live_provider_unconfigured")
        self.assertFalse(provider.health().configured)

    def test_placeholder_or_non_ascii_key_is_treated_as_unconfigured(self):
        for placeholder in ("ضع المفتاح هنا", "  ", "key with spaces"):
            transport = FakeTransport(sample_payload())
            provider = TwelveDataLiveProvider(api_key=placeholder, transport=transport)
            self.assertFalse(provider.configured, placeholder)
            with self.assertRaises(LiveProviderError) as raised:
                provider.get_ohlc(
                    "BTC/USD",
                    "M5",
                    datetime(2026, 9, 24, tzinfo=timezone.utc),
                    datetime(2026, 9, 25, tzinfo=timezone.utc),
                )
            self.assertEqual(raised.exception.code, "live_provider_unconfigured")
            self.assertEqual(transport.calls, 0)

    def test_configured_provider_reports_ready_before_first_request(self):
        provider = TwelveDataLiveProvider(api_key="secret", transport=FakeTransport(sample_payload()))
        self.assertEqual(provider.health().status, "ready")
        self.assertEqual(TwelveDataLiveProvider(api_key=None).health().status, "unconfigured")

    def test_maps_symbol_interval_and_normalizes_response(self):
        transport = FakeTransport(sample_payload())
        provider = TwelveDataLiveProvider(api_key="secret-value", transport=transport, cache_ttl_seconds=0)
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=4)
        bars = provider.get_ohlc("XAUUSD", "M5", start, end)
        self.assertEqual(len(bars), 5)
        self.assertEqual(bars[0].timestamp.tzinfo, timezone.utc)
        self.assertEqual(bars[0].open, Decimal("2500.00"))
        self.assertEqual(bars[-1].close, Decimal("2502.50"))
        request = transport.requests[0][0]
        self.assertIn("symbol=XAU%2FUSD", request.full_url)
        self.assertIn("interval=5min", request.full_url)
        self.assertNotIn("secret-value", request.full_url)
        self.assertEqual(request.headers["Authorization"], "apikey secret-value")

    def test_transient_network_failure_retries(self):
        transport = FakeTransport(sample_payload(), failures=1)
        provider = TwelveDataLiveProvider(
            api_key="secret",
            transport=transport,
            max_retries=2,
            backoff_seconds=0,
            cache_ttl_seconds=0,
        )
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        bars = provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=4))
        self.assertEqual(len(bars), 5)
        self.assertEqual(transport.calls, 2)
        self.assertEqual(provider.health().status, "healthy")

    def test_cache_prevents_duplicate_request(self):
        transport = FakeTransport(sample_payload())
        provider = TwelveDataLiveProvider(api_key="secret", transport=transport, cache_ttl_seconds=60)
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=4)
        provider.get_ohlc("EURUSD", "M1", start, end)
        provider.get_ohlc("EURUSD", "M1", start, end)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(provider.health().cache_hits, 1)


    def test_local_rate_limit_is_enforced_before_provider_call(self):
        transport = FakeTransport(sample_payload())
        provider = TwelveDataLiveProvider(
            api_key="secret",
            transport=transport,
            rate_limit_per_minute=1,
            cache_ttl_seconds=0,
        )
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=4))
        with self.assertRaises(LiveProviderError) as raised:
            provider.get_ohlc("EURUSD", "M5", start, start + timedelta(minutes=20))
        self.assertEqual(raised.exception.code, "provider_local_rate_limited")
        self.assertEqual(transport.calls, 1)

    def test_bulk_history_request_uses_a_larger_page_only_when_asked(self):
        transport = FakeTransport(sample_payload())
        provider = TwelveDataLiveProvider(api_key="secret", transport=transport, max_bars=800, cache_ttl_seconds=0)
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=4))
        provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=4), max_bars=provider.bulk_max_bars)
        self.assertIn("outputsize=800", transport.requests[0][0].full_url)
        self.assertIn("outputsize=5000", transport.requests[1][0].full_url)

    def test_local_rate_limit_waits_for_a_slot_that_frees_soon(self):
        import time as _time

        transport = FakeTransport(sample_payload())
        provider = TwelveDataLiveProvider(api_key="secret", transport=transport, rate_limit_per_minute=1,
                                          cache_ttl_seconds=0, rate_limit_wait_seconds=5)
        provider._request_times.append(_time.monotonic() - 59.8)  # frees in 0.2 s
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        began = _time.monotonic()
        provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=4))
        self.assertGreaterEqual(_time.monotonic() - began, 0.15)
        self.assertEqual(transport.calls, 1)
        # a slot that frees only after the allowed wait is still refused at once
        with self.assertRaises(LiveProviderError) as raised:
            provider.get_ohlc("EURUSD", "M5", start, start + timedelta(minutes=20))
        self.assertEqual(raised.exception.code, "provider_local_rate_limited")

    def test_provider_failure_updates_health_without_leaking_details(self):
        transport = FakeTransport({"status": "error", "code": 401, "message": "bad key"})
        provider = TwelveDataLiveProvider(api_key="secret", transport=transport, max_retries=0)
        start = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        with self.assertRaises(LiveProviderError) as raised:
            provider.get_ohlc("EURUSD", "M1", start, start + timedelta(minutes=1))
        self.assertEqual(raised.exception.code, "provider_401")
        health = provider.health().to_dict()
        self.assertEqual(health["status"], "unavailable")
        self.assertNotIn("secret", json.dumps(health))


if __name__ == "__main__":
    unittest.main()

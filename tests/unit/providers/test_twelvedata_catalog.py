from __future__ import annotations

import json
import unittest
from urllib.request import Request

from app.providers.twelvedata import TwelveDataLiveProvider


class TwelveDataCatalogTests(unittest.TestCase):
    def test_catalog_collects_forex_crypto_and_only_metal_commodities(self) -> None:
        payloads = {
            "/forex_pairs": {
                "status": "ok",
                "data": [{"symbol": "EUR/USD", "currency_group": "Major", "currency_base": "Euro", "currency_quote": "US Dollar"}],
            },
            "/cryptocurrencies": {
                "status": "ok",
                "data": [{"symbol": "BTC/USD", "currency_base": "Bitcoin", "currency_quote": "US Dollar"}],
            },
            "/commodities": {
                "status": "ok",
                "data": [
                    {"symbol": "XAU/USD", "name": "Gold Spot", "category": "Precious Metal"},
                    {"symbol": "WTI/USD", "name": "Crude Oil WTI Spot", "category": "Energy Resource"},
                ],
            },
        }

        def transport(request: Request, timeout: float) -> bytes:
            for key, payload in payloads.items():
                if key in request.full_url:
                    return json.dumps(payload).encode("utf-8")
            raise AssertionError(request.full_url)

        provider = TwelveDataLiveProvider(api_key="test", transport=transport, rate_limit_per_minute=20)
        catalog = provider.get_symbol_catalog()
        self.assertEqual([item["symbol"] for item in catalog], ["EUR/USD", "BTC/USD", "XAU/USD"])
        self.assertEqual([item["category"] for item in catalog], ["forex", "crypto", "metals"])

    def test_catalog_is_cached(self) -> None:
        calls = {"count": 0}

        def transport(request: Request, timeout: float) -> bytes:
            calls["count"] += 1
            if "/forex_pairs" in request.full_url:
                payload = {"status": "ok", "data": [{"symbol": "EUR/USD", "currency_base": "Euro", "currency_quote": "US Dollar"}]}
            elif "/cryptocurrencies" in request.full_url:
                payload = {"status": "ok", "data": [{"symbol": "BTC/USD", "currency_base": "Bitcoin"}]}
            else:
                payload = {"status": "ok", "data": [{"symbol": "XAU/USD", "name": "Gold Spot", "category": "Precious Metal"}]}
            return json.dumps(payload).encode("utf-8")

        provider = TwelveDataLiveProvider(api_key="test", transport=transport, rate_limit_per_minute=20)
        provider.get_symbol_catalog()
        provider.get_symbol_catalog()
        self.assertEqual(calls["count"], 3)


if __name__ == "__main__":
    unittest.main()

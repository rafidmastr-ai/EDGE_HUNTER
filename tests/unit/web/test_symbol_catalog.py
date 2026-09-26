from __future__ import annotations

import unittest

from app.providers.models import LiveProviderError, LiveProviderHealth
from app.web.symbol_catalog import SymbolCatalogService


class FakeCatalogProvider:
    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def get_symbol_catalog(self) -> list[dict[str, str]]:
        self.calls += 1
        return [
            {"symbol": "EUR/USD", "category": "forex", "name": "Euro / US Dollar", "group": "Major"},
            {"symbol": "GBP/USD", "category": "forex", "name": "British Pound / US Dollar", "group": "Major"},
            {"symbol": "BTC/USD", "category": "crypto", "name": "Bitcoin / US Dollar"},
            {"symbol": "XAU/USD", "category": "metals", "name": "Gold Spot", "group": "Precious Metal"},
        ]

    def health(self) -> LiveProviderHealth:
        return LiveProviderHealth(provider=self.name, configured=True, status="healthy")


class UnconfiguredCatalogProvider(FakeCatalogProvider):
    def get_symbol_catalog(self) -> list[dict[str, str]]:
        self.calls += 1
        raise LiveProviderError("not configured", code="live_provider_unconfigured", status_code=503)

    def health(self) -> LiveProviderHealth:
        return LiveProviderHealth(provider=self.name, configured=False, status="unconfigured")


class SymbolCatalogTests(unittest.TestCase):
    def test_prefix_search_prioritizes_matching_symbol(self) -> None:
        provider = FakeCatalogProvider()
        service = SymbolCatalogService(provider, cache_ttl_seconds=3600, max_results=20)
        result = service.search("eur")
        self.assertEqual(result["results"][0]["symbol"], "EUR/USD")
        self.assertEqual(result["results"][0]["category"], "forex")
        self.assertEqual(provider.calls, 1)

    def test_cache_avoids_repeated_provider_catalog_calls(self) -> None:
        provider = FakeCatalogProvider()
        service = SymbolCatalogService(provider, cache_ttl_seconds=3600, max_results=20)
        service.search("eur")
        service.search("gbp")
        self.assertEqual(provider.calls, 1)

    def test_category_filter_returns_only_requested_asset_class(self) -> None:
        service = SymbolCatalogService(FakeCatalogProvider(), cache_ttl_seconds=3600, max_results=20)
        result = service.search("", category="metals")
        self.assertEqual([item["symbol"] for item in result["results"]], ["XAU/USD"])

    def test_unconfigured_provider_uses_safe_fallback_catalog(self) -> None:
        service = SymbolCatalogService(UnconfiguredCatalogProvider(), cache_ttl_seconds=3600, max_results=20)
        result = service.search("btc")
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["results"][0]["symbol"], "BTC/USD")


if __name__ == "__main__":
    unittest.main()

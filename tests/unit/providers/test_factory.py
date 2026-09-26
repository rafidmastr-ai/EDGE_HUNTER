from __future__ import annotations

import unittest
from dataclasses import replace

from app.providers.factory import build_live_provider
from app.providers.live_provider import UnavailableLiveMarketDataProvider
from app.providers.twelvedata import TwelveDataLiveProvider
from config.config_hunter import load_settings


class LiveProviderFactoryTests(unittest.TestCase):
    def test_disabled_provider_is_safe_null_provider(self):
        settings = replace(load_settings(), live_provider_enabled=False)
        provider = build_live_provider(settings)
        self.assertIsInstance(provider, UnavailableLiveMarketDataProvider)

    def test_twelvedata_provider_builds_without_exposing_key(self):
        settings = replace(
            load_settings(),
            live_provider_enabled=True,
            live_provider_name="twelvedata",
            live_provider_api_key="test-secret",
        )
        provider = build_live_provider(settings)
        self.assertIsInstance(provider, TwelveDataLiveProvider)
        self.assertTrue(provider.configured)
        self.assertNotIn("test-secret", repr(provider.health().to_dict()))

    def test_unknown_provider_rejected(self):
        settings = replace(
            load_settings(),
            live_provider_enabled=True,
            live_provider_name="unknown",
        )
        with self.assertRaises(ValueError):
            build_live_provider(settings)


if __name__ == "__main__":
    unittest.main()

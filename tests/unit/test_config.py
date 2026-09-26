import os
import unittest
from unittest import mock

from config.config_hunter import load_settings


def _clean_environment(**overrides: str) -> dict[str, str]:
    """Process environment without any EDGE_HUNTER_* values from a local .env.

    These tests check code defaults, so they must not depend on how the
    developer's .env is configured (for example live mode with a real key).
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("EDGE_HUNTER_")}
    env.update(overrides)
    return env


class ConfigTests(unittest.TestCase):

    def test_disabled_live_provider_does_not_expose_provider_url(self):
        with mock.patch.dict(
            os.environ,
            _clean_environment(
                EDGE_HUNTER_DATA_MODE="local",
                EDGE_HUNTER_LIVE_PROVIDER_ENABLED="false",
                EDGE_HUNTER_LIVE_PROVIDER_URL="https://api.twelvedata.com",
            ),
            clear=True,
        ):
            settings = load_settings()
        self.assertFalse(settings.live_provider_enabled)
        self.assertIsNone(settings.live_provider_url)

    def test_defaults_are_safe_for_local_development(self):
        with mock.patch.dict(os.environ, _clean_environment(), clear=True):
            settings = load_settings()
        self.assertEqual(settings.environment, "development")
        self.assertEqual(settings.api_host, "127.0.0.1")
        self.assertIsNone(settings.live_provider_url)
        self.assertFalse(settings.live_provider_enabled)
        self.assertIsNone(settings.live_provider_api_key)

    def test_live_analysis_is_the_default_data_mode(self):
        with mock.patch.dict(os.environ, _clean_environment(), clear=True):
            settings = load_settings()
        self.assertEqual(settings.data_mode, "live")
        self.assertFalse(hasattr(settings, "live_fallback_to_local"))

    def test_live_mode_exposes_configured_provider_url(self):
        with mock.patch.dict(
            os.environ,
            _clean_environment(
                EDGE_HUNTER_DATA_MODE="live",
                EDGE_HUNTER_LIVE_PROVIDER="twelvedata",
                EDGE_HUNTER_LIVE_PROVIDER_ENABLED="true",
                EDGE_HUNTER_LIVE_PROVIDER_URL="https://api.twelvedata.com",
            ),
            clear=True,
        ):
            settings = load_settings()
        self.assertEqual(settings.data_mode, "live")
        self.assertTrue(settings.live_provider_enabled)
        self.assertEqual(settings.live_provider_url, "https://api.twelvedata.com")

    def test_explicit_local_mode_is_still_supported(self):
        with mock.patch.dict(os.environ, _clean_environment(EDGE_HUNTER_DATA_MODE="local"), clear=True):
            settings = load_settings()
        self.assertEqual(settings.data_mode, "local")


if __name__ == "__main__":
    unittest.main()

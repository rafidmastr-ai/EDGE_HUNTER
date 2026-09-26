import unittest

from config.config_hunter import load_settings


class ConfigTests(unittest.TestCase):

    def test_disabled_live_provider_does_not_expose_provider_url(self):
        settings = load_settings()
        self.assertFalse(settings.live_provider_enabled)
        self.assertIsNone(settings.live_provider_url)

    def test_defaults_are_safe_for_local_development(self):
        settings = load_settings()
        self.assertEqual(settings.environment, "development")
        self.assertEqual(settings.api_host, "127.0.0.1")
        self.assertIsNone(settings.live_provider_url)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from tools.production_readiness_audit import is_placeholder_value, is_safe_credential_name


class ProductionReadinessAuditHelperTests(unittest.TestCase):
    def test_password_scheme_is_not_a_credential_name(self) -> None:
        self.assertTrue(is_safe_credential_name("PASSWORD_SCHEME"))
        self.assertTrue(is_safe_credential_name("HASH_ALGORITHM"))

    def test_explicit_ascii_provider_placeholder_is_safe(self) -> None:
        self.assertTrue(is_placeholder_value("YOUR_TWELVE_DATA_API_KEY"))

    def test_arabic_provider_placeholder_is_safe(self) -> None:
        self.assertTrue(is_placeholder_value("ضع مفتاحك هنا"))

    def test_unicode_directional_marks_do_not_break_placeholder_detection(self) -> None:
        self.assertTrue(is_placeholder_value("\u200fضع مفتاحك هنا\u200e"))

    def test_realistic_key_shaped_value_is_not_a_placeholder(self) -> None:
        self.assertFalse(is_placeholder_value("td_live_1234567890abcdef"))


if __name__ == "__main__":
    unittest.main()

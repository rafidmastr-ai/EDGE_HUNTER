from __future__ import annotations

import unittest

from app.auth.security import (
    generate_subscription_code,
    hash_password,
    normalize_subscription_code,
    token_hash,
    verify_password,
)


class SecurityTests(unittest.TestCase):
    def test_password_is_hashed_and_verifies(self) -> None:
        encoded = hash_password("CorrectHorseBatteryStaple!", iterations=100_000)
        self.assertNotIn("CorrectHorseBatteryStaple!", encoded)
        self.assertTrue(verify_password("CorrectHorseBatteryStaple!", encoded))
        self.assertFalse(verify_password("wrong-password", encoded))

    def test_subscription_codes_are_exactly_16_characters(self) -> None:
        code = generate_subscription_code()
        self.assertEqual(len(code), 16)
        self.assertEqual(normalize_subscription_code(code), code)
        self.assertEqual(len(token_hash(code)), 64)

    def test_invalid_code_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_subscription_code("0123456789ABCDEF")


if __name__ == "__main__":
    unittest.main()

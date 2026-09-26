from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.auth.rate_limit import RateLimiter
from app.web.security import client_ip


class Phase11SecurityUnitTests(unittest.TestCase):
    def test_proxy_headers_are_ignored_by_default(self) -> None:
        scope = {"type": "http", "client": ("10.0.0.5", 1234), "headers": [(b"x-forwarded-for", b"203.0.113.10")]}
        self.assertEqual(client_ip(scope, trust_proxy_headers=False), "10.0.0.5")

    def test_proxy_headers_can_be_used_only_when_explicitly_enabled(self) -> None:
        scope = {"type": "http", "client": ("127.0.0.1", 1234), "headers": [(b"x-forwarded-for", b"203.0.113.10")]}
        self.assertEqual(client_ip(scope, trust_proxy_headers=True), "203.0.113.10")

    def test_rate_limiter_remains_deterministic(self) -> None:
        limiter = RateLimiter()
        limiter.check("test", limit=2, window_seconds=60)
        limiter.check("test", limit=2, window_seconds=60)
        with self.assertRaises(Exception) as ctx:
            limiter.check("test", limit=2, window_seconds=60)
        self.assertEqual(getattr(ctx.exception, "code", None), "rate_limited")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from app.providers.models import LiveProviderHealth


class LiveProviderModelTests(unittest.TestCase):
    def test_health_serialization_is_safe_and_utc(self):
        now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        health = LiveProviderHealth(
            provider="twelvedata",
            configured=True,
            status="healthy",
            last_success_at=now,
            requests_total=3,
        )
        payload = health.to_dict()
        self.assertEqual(payload["provider"], "twelvedata")
        self.assertEqual(payload["last_success_at"], "2026-09-24T12:00:00+00:00")
        self.assertNotIn("api_key", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()

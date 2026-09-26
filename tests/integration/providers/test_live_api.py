from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.db.database import Database
from app.web.app import create_web_app
from config.config_hunter import load_settings


class LiveProviderApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        settings = replace(
            load_settings(),
            database_path=root / "edge.db",
            environment="test",
            debug=False,
            allowed_hosts=("testserver",),
            docs_enabled=False,
            live_provider_enabled=False,
            live_provider_name="none",
            data_mode="live",
            log_dir=root / "logs",
        )
        self.db = Database(settings.database_path)
        self.app = create_web_app(root / "raw", database=self.db, settings=settings)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_live_health_does_not_leak_configuration_secrets(self):
        response = self.client.get("/api/live-health")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["provider"], "none")
        self.assertFalse(body["configured"])
        self.assertNotIn("api_key", body)

    def test_live_mode_returns_controlled_503_when_provider_unavailable(self):
        csrf = self.client.get("/api/auth/csrf").json()["csrf_token"]
        register = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": csrf},
            json={
                "email": "live@example.com",
                "password": "StrongPass!123",
                "password_confirm": "StrongPass!123",
            },
        )
        self.assertEqual(register.status_code, 200, register.text)
        csrf = register.json()["csrf_token"]
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": csrf},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"]["code"], "live_provider_unconfigured")
        self.assertEqual(response.json()["detail"]["message"], "live market data is currently unavailable")


if __name__ == "__main__":
    unittest.main()

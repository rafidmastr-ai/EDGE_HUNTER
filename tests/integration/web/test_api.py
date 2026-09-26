from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.db.database import Database
from app.web.app import create_web_app
from app.web.schemas import AnalyzeRequest
from config.config_hunter import load_settings


class WebAPIIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        price = 2300.0
        with (root / "XAUUSD.csv").open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close\n")
            for i in range(900):
                ts = start + timedelta(minutes=i)
                op = price
                cl = price + (0.3 if i % 2 else -0.15)
                hi = max(op, cl) + 0.5
                lo = min(op, cl) - 0.45
                handle.write(f"{ts.isoformat()},{op},{hi},{lo},{cl}\n")
                price = cl
        settings = replace(load_settings(), database_path=root / "edge.db", password_iterations=100_000, cookie_secure=False, data_mode="local", live_provider_enabled=False)
        self.database = Database(settings.database_path)
        self.client = TestClient(create_web_app(root, database=self.database, settings=settings))
        self.csrf = self.client.get("/api/auth/csrf").json()["csrf_token"]
        response = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": self.csrf},
            json={"email": "web@example.com", "password": "StrongPass!123", "password_confirm": "StrongPass!123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.csrf = response.json()["csrf_token"]

    def tearDown(self) -> None:
        self.database.close()
        self.tmp.cleanup()

    def test_public_auth_status_bootstrap_does_not_require_login(self) -> None:
        other_settings = load_settings()
        other_settings = replace(other_settings, database_path=Path(self.tmp.name) / "status.db", cookie_secure=False, data_mode="local", live_provider_enabled=False)
        other_db = Database(other_settings.database_path)
        try:
            anonymous = TestClient(create_web_app(Path(self.tmp.name), database=other_db, settings=other_settings))
            response = anonymous.get("/api/auth/status")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(response.json()["authenticated"])
            self.assertEqual(response.json()["reason"], "anonymous")
            self.assertIsNone(response.json()["user"])
        finally:
            other_db.close()

        response = self.client.get("/api/auth/status")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["authenticated"])
        self.assertEqual(body["reason"], "authenticated")
        self.assertEqual(body["user"]["email"], "web@example.com")

    def test_health_and_meta(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        meta = self.client.get("/api/meta")
        self.assertEqual(meta.status_code, 200)
        self.assertIn("XAUUSD", meta.json()["symbols"])
        self.assertTrue(meta.json()["auth_required_for_analysis"])


    def test_symbol_catalog_search_is_public_and_supports_prefix_matching(self) -> None:
        response = self.client.get("/api/symbols/search?q=xau&category=metals")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["category"], "metals")
        self.assertTrue(payload["results"])
        self.assertEqual(payload["results"][0]["symbol"], "XAU/USD")

    def test_dynamic_symbol_format_accepts_forex_and_crypto_provider_symbols(self) -> None:
        self.assertEqual(
            AnalyzeRequest(symbol="GBP/USD").symbol,
            "GBP/USD",
        )
        self.assertEqual(
            AnalyzeRequest(symbol="BTC/USD").symbol,
            "BTC/USD",
        )

    def test_analyze_returns_api_ready_payload(self) -> None:
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        for key in ("direction", "confidence", "entry", "target", "stop_loss", "lot_size", "strategy_comparison", "chart"):
            self.assertIn(key, payload)

    def test_missing_dataset_maps_to_data_unavailable(self) -> None:
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "EURUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "data_unavailable")

    def test_invalid_risk_is_rejected(self) -> None:
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "XAUUSD", "risk_percent": 0, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 422)

    def test_analysis_requires_authentication(self) -> None:
        other_settings = load_settings()
        other_settings = replace(other_settings, database_path=Path(self.tmp.name) / "other.db", cookie_secure=False, data_mode="local", live_provider_enabled=False)
        other_db = Database(other_settings.database_path)
        try:
            anonymous = TestClient(create_web_app(Path(self.tmp.name), database=other_db, settings=other_settings))
            response = anonymous.post("/api/analyze", json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["detail"]["code"], "unauthorized")
        finally:
            other_db.close()


if __name__ == "__main__":
    unittest.main()

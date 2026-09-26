from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.repository import utc_now
from app.auth.security import hash_password
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.web.app import create_web_app
from config.config_hunter import load_settings


class Phase12FullSystemIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_root = root / "raw"
        self.data_root.mkdir()
        self._write_dataset(self.data_root / "XAUUSD.csv")
        settings = replace(
            load_settings(),
            database_path=root / "edge.db",
            password_iterations=100_000,
            session_ttl_hours=24,
            login_rate_limit=20,
            rate_limit_window_seconds=60,
            public_rate_limit=200,
            public_rate_limit_window_seconds=60,
            allowed_hosts=("testserver", "127.0.0.1"),
            cors_origins=("https://app.example.com",),
            cookie_secure=False,
            docs_enabled=True,
            environment="test",
            data_mode="local",
            live_provider_enabled=False,
            debug=False,
            trust_proxy_headers=False,
            max_request_bytes=65_536,
            log_dir=root / "logs",
        )
        self.db = Database(settings.database_path)
        self.app = create_web_app(self.data_root, database=self.db, settings=settings)
        self.client = TestClient(self.app)
        self.csrf = self._bootstrap_csrf()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    @staticmethod
    def _write_dataset(path: Path) -> None:
        from datetime import datetime, timezone

        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        price = 2400.0
        with path.open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close,volume\n")
            for i in range(720):
                ts = start + timedelta(minutes=i)
                cycle = i % 12
                delta = 0.35 if cycle < 7 else -0.20
                op = price
                cl = price + delta
                hi = max(op, cl) + 0.55
                lo = min(op, cl) - 0.45
                handle.write(f"{ts.isoformat()},{op},{hi},{lo},{cl},{100+i}\n")
                price = cl

    def _bootstrap_csrf(self) -> str:
        response = self.client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["csrf_token"]

    def _register(self, email: str = "journey@example.com") -> dict:
        response = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": self.csrf},
            json={
                "email": email,
                "password": "StrongPass!123",
                "password_confirm": "StrongPass!123",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.csrf = body["csrf_token"]
        return body

    def _create_admin(self) -> None:
        now = utc_now().isoformat()
        password_hash = hash_password("AdminPass!123", iterations=100_000)
        cursor = self.db.execute(
            """
            INSERT INTO users(
                email, password_hash, role, status,
                trial_started_at, trial_expires_at, created_at
            ) VALUES (?, ?, 'admin', 'active', ?, ?, ?)
            """,
            ("admin@example.com", password_hash, now, now, now),
        )
        self.db.commit()
        self.admin_user_id = int(cursor.lastrowid)

    def _admin_client(self) -> tuple[TestClient, str]:
        self._create_admin()
        settings = self.app.state.settings
        admin_app = create_web_app(self.data_root, database=self.db, settings=settings)
        client = TestClient(admin_app)
        csrf = client.get("/api/auth/csrf").json()["csrf_token"]
        login = client.post(
            "/api/admin/login",
            headers={"X-CSRF-Token": csrf},
            json={"email": "admin@example.com", "password": "AdminPass!123"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        return client, login.json()["csrf_token"]

    def test_complete_user_journey_trial_expiry_subscription_restore(self) -> None:
        # register -> trial
        registration = self._register()
        self.assertTrue(registration["user"]["access"]["allowed"])
        self.assertTrue(registration["user"]["access"]["trial_active"])

        # choose symbol/risk + optional capital -> analyze -> signal/chart/reasons
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "capital": 10_000, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["symbol"], "XAUUSD")
        self.assertIn(result["direction"], {"BUY", "SELL", "NO_CLEAR_SIGNAL"})
        self.assertIn("chart", result)
        self.assertIn("reasons", result)
        if result["direction"] != "NO_CLEAR_SIGNAL":
            self.assertIsNotNone(result["entry"])
            self.assertIsNotNone(result["target"])
            self.assertIsNotNone(result["stop_loss"])
        if result["direction"] != "NO_CLEAR_SIGNAL":
            self.assertIsNotNone(result["capital_impact"])
        else:
            self.assertIsNone(result["capital_impact"])

        # frontend copy contract is covered separately; verify API exposes one target only.
        self.assertNotIn("targets", result)
        self.assertIsNone(result["metadata"].get("targets"))

        # admin login -> generate one subscription code
        admin_client, admin_csrf = self._admin_client()
        generated = admin_client.post(
            "/api/admin/codes",
            headers={"X-CSRF-Token": admin_csrf},
            json={"duration_months": 1, "quantity": 1},
        )
        self.assertEqual(generated.status_code, 200, generated.text)
        code = generated.json()["codes"][0]["code"]
        self.assertEqual(len(code), 16)

        # expire trial on purpose -> analysis must be blocked server-side
        user_row = self.db.execute(
            "SELECT id FROM users WHERE email = ?", ("journey@example.com",)
        ).fetchone()
        self.assertIsNotNone(user_row)
        expired_at = (utc_now() - timedelta(minutes=1)).isoformat()
        self.db.execute(
            "UPDATE users SET trial_expires_at = ? WHERE id = ?",
            (expired_at, int(user_row["id"])),
        )
        self.db.commit()

        blocked = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(blocked.status_code, 403, blocked.text)
        self.assertEqual(blocked.json()["detail"]["code"], "subscription_expired")

        # subscription code -> access restored -> analyze again
        redeemed = self.client.post(
            "/api/auth/redeem",
            headers={"X-CSRF-Token": self.csrf},
            json={"code": code},
        )
        self.assertEqual(redeemed.status_code, 200, redeemed.text)
        self.assertTrue(redeemed.json()["allowed"])
        self.assertTrue(redeemed.json()["subscription_active"])

        restored = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf},
            json={"symbol": "XAUUSD", "risk_percent": 0.5, "lot_mode": "manual", "lot_size": 0.01},
        )
        self.assertEqual(restored.status_code, 200, restored.text)

        dashboard = admin_client.get("/api/admin/dashboard")
        self.assertEqual(dashboard.status_code, 200, dashboard.text)
        self.assertGreaterEqual(dashboard.json()["counts"]["subscription_active"], 1)

    def test_database_migrations_are_idempotent_and_core_tables_exist(self) -> None:
        MigrationRunner(self.db).apply_all()
        MigrationRunner(self.db).apply_all()
        rows = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?, ?, ?, ?, ?) ORDER BY name",
            ("schema_migrations", "app_metadata", "users", "trial_devices", "subscriptions", "subscription_codes", "auth_sessions"),
        ).fetchall()
        names = {row["name"] for row in rows}
        self.assertEqual(
            names,
            {"schema_migrations", "app_metadata", "users", "trial_devices", "subscriptions", "subscription_codes", "auth_sessions"},
        )

    def test_database_file_is_not_publicly_served(self) -> None:
        response = self.client.get("/data/edge_hunter.db")
        self.assertIn(response.status_code, {404, 400})
        self.assertNotIn("SQLite format", response.text)

    def test_anonymous_client_cannot_analyze_or_admin(self) -> None:
        anonymous = TestClient(self.app)
        analyze = anonymous.post(
            "/api/analyze",
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(analyze.status_code, 401)
        admin = anonymous.get("/api/admin/dashboard")
        self.assertEqual(admin.status_code, 401)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.repository import utc_now
from app.auth.security import hash_password, token_hash
from app.db.database import Database
from app.web.app import create_web_app
from config.config_hunter import load_settings


class AuthFlowTests(unittest.TestCase):
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
            session_ttl_hours=1,
            login_rate_limit=5,
            rate_limit_window_seconds=300,
            cookie_secure=False,
        )
        self.database = Database(settings.database_path)
        self.client = TestClient(create_web_app(self.data_root, database=self.database, settings=settings))
        self._csrf()

    def tearDown(self) -> None:
        self.database.close()
        self.tmp.cleanup()

    @staticmethod
    def _write_dataset(path: Path) -> None:
        from datetime import datetime, timezone

        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        price = 2500.0
        with path.open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close\n")
            for i in range(300):
                ts = start + timedelta(minutes=i)
                op = price
                cl = price + (0.4 if i % 2 else -0.2)
                hi = max(op, cl) + 0.6
                lo = min(op, cl) - 0.5
                handle.write(f"{ts.isoformat()},{op},{hi},{lo},{cl}\n")
                price = cl

    def _csrf(self) -> str:
        response = self.client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200)
        token = response.json()["csrf_token"]
        self.csrf_token = token
        return token

    def _register(self, email: str = "user@example.com") -> dict:
        response = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": self.csrf_token},
            json={"email": email, "password": "StrongPass!123", "password_confirm": "StrongPass!123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.csrf_token = body["csrf_token"]
        return body

    def _create_admin(self, email: str = "admin@example.com", password: str = "AdminPass!123") -> int:
        now = utc_now().isoformat()
        password_hash = hash_password(password, iterations=100_000)
        cursor = self.database.execute(
            """
            INSERT INTO users(email, password_hash, role, status, trial_started_at, trial_expires_at, created_at)
            VALUES (?, ?, 'admin', 'active', ?, ?, ?)
            """,
            (email, password_hash, now, now, now),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def _admin_login(self) -> str:
        self._create_admin()
        admin_client = TestClient(create_web_app(self.data_root, database=self.database, settings=replace(load_settings(), database_path=self.database.path, password_iterations=100_000, session_ttl_hours=1, cookie_secure=False)))
        boot = admin_client.get("/api/auth/csrf").json()["csrf_token"]
        response = admin_client.post(
            "/api/admin/login",
            headers={"X-CSRF-Token": boot},
            json={"email": "admin@example.com", "password": "AdminPass!123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return admin_client, response.json()["csrf_token"]

    def test_auth_cookies_are_http_only_and_same_site(self) -> None:
        response = self.client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": self.csrf_token},
            json={"email": "cookies@example.com", "password": "StrongPass!123", "password_confirm": "StrongPass!123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        set_cookie = "\n".join(response.headers.get_list("set-cookie"))
        self.assertIn("eh_session=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=lax", set_cookie)
        self.assertIn("eh_device=", set_cookie)

    def test_registration_creates_seven_day_trial_and_allows_me(self) -> None:
        body = self._register()
        self.assertTrue(body["user"]["access"]["allowed"])
        self.assertTrue(body["user"]["access"]["trial_active"])
        self.assertFalse(body["user"]["access"]["subscription_active"])
        self.assertEqual(body["user"]["access"]["label_ar"], "الفترة التجريبية فعالة")

    def test_logout_then_login_restores_same_device(self) -> None:
        self._register()
        logout = self.client.post("/api/auth/logout", headers={"X-CSRF-Token": self.csrf_token})
        self.assertEqual(logout.status_code, 200)
        self._csrf()
        login = self.client.post(
            "/api/auth/login",
            headers={"X-CSRF-Token": self.csrf_token},
            json={"email": "user@example.com", "password": "StrongPass!123"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.csrf_token = login.json()["csrf_token"]

    def test_expired_trial_blocks_analysis_server_side(self) -> None:
        self._register()
        user_id = self.client.get("/api/auth/me").json()["user"]["id"]
        past = (utc_now() - timedelta(days=1)).isoformat()
        self.database.execute("UPDATE users SET trial_expires_at = ? WHERE id = ?", (past, user_id))
        self.database.commit()
        response = self.client.post(
            "/api/analyze",
            headers={"X-CSRF-Token": self.csrf_token},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "subscription_expired")

    def test_invalid_used_and_expired_code_paths(self) -> None:
        self._register()
        invalid = self.client.post("/api/auth/redeem", headers={"X-CSRF-Token": self.csrf_token}, json={"code": "A" * 16})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["detail"]["code"], "invalid_code")

        admin_client, admin_csrf = self._admin_login()
        generated = admin_client.post(
            "/api/admin/codes",
            headers={"X-CSRF-Token": admin_csrf},
            json={"duration_months": 1, "quantity": 1},
        )
        self.assertEqual(generated.status_code, 200, generated.text)
        code = generated.json()["codes"][0]["code"]

        redeemed = self.client.post("/api/auth/redeem", headers={"X-CSRF-Token": self.csrf_token}, json={"code": code})
        self.assertEqual(redeemed.status_code, 200, redeemed.text)
        used = self.client.post("/api/auth/redeem", headers={"X-CSRF-Token": self.csrf_token}, json={"code": code})
        self.assertEqual(used.status_code, 422)
        self.assertEqual(used.json()["detail"]["code"], "code_used")

        expired = "B" * 16
        self.database.execute(
            """
            INSERT INTO subscription_codes(code_hash, code_hint, duration_months, status, expires_at, created_at)
            VALUES (?, ?, 1, 'available', ?, ?)
            """,
            (token_hash(expired), "BBBB", (utc_now() - timedelta(days=1)).isoformat(), utc_now().isoformat()),
        )
        self.database.commit()
        expired_response = self.client.post("/api/auth/redeem", headers={"X-CSRF-Token": self.csrf_token}, json={"code": expired})
        self.assertEqual(expired_response.status_code, 422)
        self.assertEqual(expired_response.json()["detail"]["code"], "code_expired")

    def test_device_binding_rejects_second_device(self) -> None:
        self._register()
        second = TestClient(create_web_app(self.data_root, database=self.database, settings=replace(load_settings(), database_path=self.database.path, password_iterations=100_000, session_ttl_hours=1, cookie_secure=False)))
        boot = second.get("/api/auth/csrf").json()["csrf_token"]
        response = second.post(
            "/api/auth/login",
            headers={"X-CSRF-Token": boot},
            json={"email": "user@example.com", "password": "StrongPass!123"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "device_mismatch")

    def test_create_codes_rejects_stale_admin_id_without_partial_insert(self) -> None:
        admin_id = self._create_admin()
        auth = self.client.app.state.auth
        before_codes = int(self.database.execute("SELECT COUNT(*) FROM subscription_codes").fetchone()[0])
        before_audit = int(self.database.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0])

        with self.assertRaises(Exception) as caught:
            auth.create_codes(admin_id + 999, 1, 1)

        self.assertEqual(getattr(caught.exception, "code", None), "admin_session_stale")
        self.assertEqual(int(self.database.execute("SELECT COUNT(*) FROM subscription_codes").fetchone()[0]), before_codes)
        self.assertEqual(int(self.database.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]), before_audit)

    def test_create_codes_is_atomic_between_code_and_audit(self) -> None:
        admin_id = self._create_admin()
        auth = self.client.app.state.auth
        generated = auth.create_codes(admin_id, 1, 1)
        self.assertEqual(len(generated), 1)
        code_row = self.database.execute("SELECT code_hash, status FROM subscription_codes ORDER BY id DESC LIMIT 1").fetchone()
        self.assertIsNotNone(code_row)
        audit_row = self.database.execute(
            "SELECT actor_user_id, action, target_type FROM audit_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(audit_row)
        self.assertEqual(int(audit_row["actor_user_id"]), admin_id)
        self.assertEqual(str(audit_row["action"]), "generate_code")
        self.assertEqual(str(audit_row["target_type"]), "subscription_code")

    def test_admin_authorization_and_management(self) -> None:
        self._register()
        user_id = self.client.get("/api/auth/me").json()["user"]["id"]
        # Expire the free trial so revoking the paid subscription leaves no other access path.
        self.database.execute("UPDATE users SET trial_expires_at = ? WHERE id = ?", ((utc_now() - timedelta(days=1)).isoformat(), user_id))
        self.database.commit()
        forbidden = self.client.get("/api/admin/dashboard")
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(forbidden.json()["detail"]["code"], "admin_forbidden")

        admin_client, admin_csrf = self._admin_login()
        dashboard = admin_client.get("/api/admin/dashboard")
        self.assertEqual(dashboard.status_code, 200)

        extended = admin_client.post(
            f"/api/admin/users/{user_id}/extend",
            headers={"X-CSRF-Token": admin_csrf},
            json={"duration_months": 1},
        )
        self.assertEqual(extended.status_code, 200, extended.text)
        self.assertTrue(extended.json()["allowed"])

        revoked = admin_client.post(
            f"/api/admin/users/{user_id}/revoke",
            headers={"X-CSRF-Token": admin_csrf},
            json={"reason": "test"},
        )
        self.assertEqual(revoked.status_code, 200, revoked.text)
        me = self.client.get("/api/auth/me").json()["user"]
        self.assertFalse(me["access"]["allowed"])

        relink = admin_client.post(f"/api/admin/users/{user_id}/relink", headers={"X-CSRF-Token": admin_csrf})
        self.assertEqual(relink.status_code, 200)
        old_me = self.client.get("/api/auth/me")
        self.assertEqual(old_me.status_code, 401)

    def test_login_rate_limit_is_enforced(self) -> None:
        for _ in range(5):
            response = self.client.post(
                "/api/auth/login",
                headers={"X-CSRF-Token": self.csrf_token},
                json={"email": "nobody@example.com", "password": "StrongPass!123"},
            )
            self.assertEqual(response.status_code, 401)
        blocked = self.client.post(
            "/api/auth/login",
            headers={"X-CSRF-Token": self.csrf_token},
            json={"email": "nobody@example.com", "password": "StrongPass!123"},
        )
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["detail"]["code"], "rate_limited")

    def test_session_expiry_is_enforced(self) -> None:
        self._register()
        token_hash_value = token_hash(self.client.cookies.get("eh_session"))
        self.database.execute(
            "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
            ((utc_now() - timedelta(minutes=1)).isoformat(), token_hash_value),
        )
        self.database.commit()
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["code"], "session_expired")

    def test_csrf_is_required_for_state_change(self) -> None:
        self._register()
        response = self.client.post("/api/auth/redeem", json={"code": "A" * 16})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "csrf_required")


if __name__ == "__main__":
    unittest.main()

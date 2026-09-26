"""User + Admin sessions in the same browser.

One ``TestClient`` keeps one cookie jar, exactly like one browser. These tests
drive the user area and the admin area through the *same* client and prove the
two sessions (and their CSRF tokens) are fully independent.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.repository import utc_now
from app.auth.security import hash_password, token_hash
from app.db.database import Database
from app.web.app import create_web_app
from config.config_hunter import load_settings

USER_EMAIL = "user@example.com"
USER_PASSWORD = "StrongPass!123"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "AdminPass!123"


class UserAdminSameBrowserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_root = root / "raw"
        self.data_root.mkdir()
        self._write_dataset(self.data_root / "XAUUSD.csv")
        self.settings = replace(
            load_settings(),
            database_path=root / "edge.db",
            log_dir=root / "logs",
            password_iterations=100_000,
            session_ttl_hours=1,
            login_rate_limit=50,
            rate_limit_window_seconds=300,
            cookie_secure=False,
            data_mode="local",
            live_provider_enabled=False,
        )
        self.database = Database(self.settings.database_path)
        self.app = create_web_app(self.data_root, database=self.database, settings=self.settings)
        self.browser = TestClient(self.app)
        self.admin_id = self._create_admin()

    def tearDown(self) -> None:
        self.browser.close()
        self.database.close()
        self.tmp.cleanup()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _write_dataset(path: Path) -> None:
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        price = 2500.0
        with path.open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close\n")
            for i in range(300):
                ts = start + timedelta(minutes=i)
                op = price
                cl = price + (0.4 if i % 2 else -0.2)
                handle.write(f"{ts.isoformat()},{op},{max(op, cl) + 0.6},{min(op, cl) - 0.5},{cl}\n")
                price = cl

    def _create_admin(self, email: str = ADMIN_EMAIL) -> int:
        now = utc_now().isoformat()
        cursor = self.database.execute(
            """
            INSERT INTO users(email, password_hash, role, status, trial_started_at, trial_expires_at, created_at)
            VALUES (?, ?, 'admin', 'active', ?, ?, ?)
            """,
            (email, hash_password(ADMIN_PASSWORD, iterations=100_000), now, now, now),
        )
        self.database.commit()
        return int(cursor.lastrowid)

    def _new_browser(self) -> TestClient:
        return TestClient(self.app)

    def _user_register(self, client: TestClient, email: str = USER_EMAIL) -> str:
        boot = client.get("/api/auth/csrf").json()["csrf_token"]
        response = client.post(
            "/api/auth/register",
            headers={"X-CSRF-Token": boot},
            json={"email": email, "password": USER_PASSWORD, "password_confirm": USER_PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["csrf_token"]

    def _user_login(self, client: TestClient, email: str = USER_EMAIL) -> str:
        boot = client.get("/api/auth/csrf").json()["csrf_token"]
        response = client.post(
            "/api/auth/login",
            headers={"X-CSRF-Token": boot},
            json={"email": email, "password": USER_PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["csrf_token"]

    def _admin_login(self, client: TestClient) -> str:
        boot = client.get("/api/admin/csrf").json()["csrf_token"]
        response = client.post(
            "/api/admin/login",
            headers={"X-CSRF-Token": boot},
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["csrf_token"]

    def _user_status(self, client: TestClient | None = None) -> dict:
        response = (client or self.browser).get("/api/auth/status")
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _admin_status(self, client: TestClient | None = None) -> dict:
        response = (client or self.browser).get("/api/admin/status")
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _generate_code(self, admin_csrf: str, client: TestClient | None = None) -> str:
        response = (client or self.browser).post(
            "/api/admin/codes",
            headers={"X-CSRF-Token": admin_csrf},
            json={"duration_months": 1, "quantity": 1},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["codes"][0]["code"]

    def _redeem(self, code: str, csrf: str, client: TestClient | None = None):
        return (client or self.browser).post("/api/auth/redeem", headers={"X-CSRF-Token": csrf}, json={"code": code})

    def _reload_user_page(self, client: TestClient | None = None) -> str | None:
        """Mimic app.js bootstrap on reload: bootstrap token + status, then the
        session CSRF token is read back from the eh_csrf cookie."""
        client = client or self.browser
        self.assertEqual(client.get("/").status_code, 200)
        client.get("/api/auth/csrf")
        self._user_status(client)
        return client.cookies.get("eh_csrf")

    def _reload_admin_page(self, client: TestClient | None = None) -> str | None:
        client = client or self.browser
        self.assertEqual(client.get("/admin").status_code, 200)
        client.get("/api/admin/csrf")
        self._admin_status(client)
        return client.cookies.get("eh_admin_csrf")

    def _count(self, table: str) -> int:
        return int(self.database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    # ------------------------------------------------------ root-cause guard
    def test_session_resolves_user_id_not_session_id(self) -> None:
        """Regression: the session lookup used ``SELECT s.*, u.…`` so the
        resolved user id was the *session* row id. Once admin logins created
        extra sessions, the user's session id no longer matched any user id and
        redeem failed with "login required" (or hit another account)."""
        for _ in range(3):
            self._admin_login(self._new_browser())
        user_csrf = self._user_register(self.browser)
        user_id = int(self.database.execute("SELECT id FROM users WHERE email = ?", (USER_EMAIL,)).fetchone()["id"])
        session_id = int(
            self.database.execute(
                "SELECT id FROM auth_sessions WHERE token_hash = ?",
                (token_hash(self.browser.cookies.get("eh_session")),),
            ).fetchone()["id"]
        )
        self.assertNotEqual(user_id, session_id)
        me = self.browser.get("/api/auth/me").json()["user"]
        self.assertEqual(me["id"], user_id)
        self.assertEqual(self._user_status()["user"]["id"], user_id)
        self.assertTrue(me["access"]["trial_active"])

        admin_csrf = self._admin_login(self.browser)
        self.assertEqual(self._admin_status()["user"]["id"], self.admin_id)
        code = self._generate_code(admin_csrf)
        audit_actor = self.database.execute(
            "SELECT actor_user_id FROM audit_logs WHERE action = 'generate_code'"
        ).fetchone()["actor_user_id"]
        self.assertEqual(int(audit_actor), self.admin_id)

        redeemed = self._redeem(code, user_csrf)
        self.assertEqual(redeemed.status_code, 200, redeemed.text)
        assigned = self.database.execute("SELECT assigned_user_id FROM subscription_codes").fetchone()["assigned_user_id"]
        self.assertEqual(int(assigned), user_id)
        subscription_owner = self.database.execute("SELECT user_id FROM subscriptions").fetchone()["user_id"]
        self.assertEqual(int(subscription_owner), user_id)

    # -------------------------------------------------------------- A / B / C
    def test_a_user_login_establishes_user_session_only(self) -> None:
        self._user_register(self.browser)
        self.assertIsNotNone(self.browser.cookies.get("eh_session"))
        self.assertIsNotNone(self.browser.cookies.get("eh_csrf"))
        self.assertIsNone(self.browser.cookies.get("eh_admin_session"))
        self.assertTrue(self._user_status()["authenticated"])
        self.assertFalse(self._admin_status()["authenticated"])

    def test_b_admin_login_establishes_admin_session_only(self) -> None:
        response_boot = self.browser.get("/api/admin/csrf")
        boot = response_boot.json()["csrf_token"]
        response = self.browser.post(
            "/api/admin/login",
            headers={"X-CSRF-Token": boot},
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.text)
        set_cookie = "\n".join(response.headers.get_list("set-cookie"))
        self.assertIn("eh_admin_session=", set_cookie)
        self.assertIn("eh_admin_csrf=", set_cookie)
        self.assertNotIn("eh_session=", set_cookie)
        self.assertNotIn("eh_csrf=", set_cookie)
        admin_cookie_line = next(line for line in response.headers.get_list("set-cookie") if line.startswith("eh_admin_session="))
        self.assertIn("HttpOnly", admin_cookie_line)
        self.assertIn("SameSite=lax", admin_cookie_line)
        self.assertIn("Path=/", admin_cookie_line)
        self.assertTrue(self._admin_status()["authenticated"])
        self.assertFalse(self._user_status()["authenticated"])
        self.assertEqual(self.browser.get("/api/admin/dashboard").status_code, 200)

    def test_c_user_and_admin_sessions_coexist_with_distinct_tokens(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        self.assertNotEqual(self.browser.cookies.get("eh_session"), self.browser.cookies.get("eh_admin_session"))
        self.assertNotEqual(user_csrf, admin_csrf)
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)
        self.assertEqual(self.browser.cookies.get("eh_admin_csrf"), admin_csrf)
        user_status = self._user_status()
        admin_status = self._admin_status()
        self.assertEqual(user_status["user"]["email"], USER_EMAIL)
        self.assertEqual(admin_status["user"]["email"], ADMIN_EMAIL)
        scopes = {
            row["scope"]
            for row in self.database.execute("SELECT scope FROM auth_sessions").fetchall()
        }
        self.assertEqual(scopes, {"user", "admin"})

    # ------------------------------------------------------ Full scenario / R
    def test_full_same_browser_scenario_generate_then_redeem(self) -> None:
        user_csrf = self._user_register(self.browser)
        user_session = self.browser.cookies.get("eh_session")
        self.assertTrue(self._user_status()["authenticated"])

        admin_csrf = self._admin_login(self.browser)
        self.assertEqual(self.browser.get("/admin").status_code, 200)
        self.assertEqual(self.browser.get("/api/admin/dashboard").status_code, 200)
        code = self._generate_code(admin_csrf)

        # User session untouched by the admin activity.
        self.assertEqual(self.browser.cookies.get("eh_session"), user_session)
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)
        self.assertEqual(self.browser.get("/").status_code, 200)
        status = self._user_status()
        self.assertTrue(status["authenticated"])
        self.assertEqual(status["user"]["email"], USER_EMAIL)

        redeemed = self._redeem(code, user_csrf)
        self.assertEqual(redeemed.status_code, 200, redeemed.text)
        self.assertTrue(redeemed.json()["subscription_active"])
        me = self.browser.get("/api/auth/me").json()["user"]
        self.assertTrue(me["access"]["subscription_active"])
        code_row = self.database.execute(
            "SELECT status, assigned_user_id FROM subscription_codes WHERE code_hash = ?",
            (token_hash(code),),
        ).fetchone()
        self.assertEqual(code_row["status"], "redeemed")
        self.assertEqual(int(code_row["assigned_user_id"]), me["id"])
        self.assertNotEqual(int(code_row["assigned_user_id"]), self.admin_id)

        # Admin logout leaves the user session fully usable.
        logout = self.browser.post("/api/admin/logout", headers={"X-CSRF-Token": admin_csrf})
        self.assertEqual(logout.status_code, 200, logout.text)
        self.assertIsNone(self.browser.cookies.get("eh_admin_session"))
        self.assertEqual(self.browser.cookies.get("eh_session"), user_session)
        self.assertFalse(self._admin_status()["authenticated"])
        self.assertTrue(self._user_status()["authenticated"])
        analysis = self.browser.post(
            "/api/analyze",
            headers={"X-CSRF-Token": user_csrf},
            json={"symbol": "XAUUSD", "risk_percent": 1.0, "lot_mode": "auto"},
        )
        self.assertNotIn(analysis.status_code, {401, 403}, analysis.text)

        # And the reverse: user logout leaves a fresh admin session alive.
        admin_csrf = self._admin_login(self.browser)
        user_logout = self.browser.post("/api/auth/logout", headers={"X-CSRF-Token": user_csrf})
        self.assertEqual(user_logout.status_code, 200, user_logout.text)
        self.assertFalse(self._user_status()["authenticated"])
        self.assertTrue(self._admin_status()["authenticated"])
        self.assertEqual(self.browser.get("/api/admin/dashboard").status_code, 200)
        self._generate_code(admin_csrf)

    # -------------------------------------------------------------- D / E / F
    def test_d_admin_login_does_not_replace_user_session(self) -> None:
        user_csrf = self._user_register(self.browser)
        before = self.browser.cookies.get("eh_session")
        self._admin_login(self.browser)
        self.assertEqual(self.browser.cookies.get("eh_session"), before)
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)
        self.assertEqual(self._user_status()["user"]["email"], USER_EMAIL)
        self.assertEqual(self.browser.get("/api/auth/me").status_code, 200)

    def test_e_admin_logout_does_not_remove_user_session(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        sessions_before = self._count("auth_sessions")
        response = self.browser.post("/api/admin/logout", headers={"X-CSRF-Token": admin_csrf})
        self.assertEqual(response.status_code, 200, response.text)
        cleared = "\n".join(response.headers.get_list("set-cookie"))
        self.assertIn("eh_admin_session=", cleared)
        self.assertNotIn("eh_session=", cleared)
        self.assertNotIn("eh_csrf=", cleared)
        self.assertEqual(self._count("auth_sessions"), sessions_before - 1)
        self.assertTrue(self._user_status()["authenticated"])
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)
        # The admin session is really gone server-side.
        self.assertEqual(self.browser.get("/api/admin/dashboard").status_code, 403)

    def test_f_user_logout_does_not_remove_admin_session(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        response = self.browser.post("/api/auth/logout", headers={"X-CSRF-Token": user_csrf})
        self.assertEqual(response.status_code, 200, response.text)
        cleared = "\n".join(response.headers.get_list("set-cookie"))
        self.assertNotIn("eh_admin_session=", cleared)
        self.assertNotIn("eh_admin_csrf=", cleared)
        self.assertFalse(self._user_status()["authenticated"])
        self.assertTrue(self._admin_status()["authenticated"])
        self.assertEqual(self.browser.cookies.get("eh_admin_csrf"), admin_csrf)
        self._generate_code(admin_csrf)

    # ------------------------------------------------------------------ G / H
    def test_g_user_redeems_after_admin_login(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)
        response = self._redeem(code, user_csrf)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "active_subscription")

    def test_h_user_redeems_after_admin_logout(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)
        self.assertEqual(self.browser.post("/api/admin/logout", headers={"X-CSRF-Token": admin_csrf}).status_code, 200)
        response = self._redeem(code, user_csrf)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["subscription_active"])

    # ---------------------------------------------------------------------- I
    def test_i_user_and_admin_csrf_tokens_are_not_interchangeable(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)

        wrong_for_user = self._redeem(code, admin_csrf)
        self.assertEqual(wrong_for_user.status_code, 403)
        self.assertEqual(wrong_for_user.json()["detail"]["code"], "csrf_invalid")
        self.assertNotIn("session_id", wrong_for_user.json()["detail"])

        wrong_for_admin = self.browser.post(
            "/api/admin/codes",
            headers={"X-CSRF-Token": user_csrf},
            json={"duration_months": 1, "quantity": 1},
        )
        self.assertEqual(wrong_for_admin.status_code, 403)
        self.assertEqual(wrong_for_admin.json()["detail"]["code"], "csrf_invalid")

        bootstrap = self.browser.get("/api/auth/csrf").json()["csrf_token"]
        self.assertEqual(self._redeem(code, bootstrap).json()["detail"]["code"], "csrf_invalid")
        # The correct tokens still work after the rejected attempts.
        self.assertEqual(self._redeem(code, user_csrf).status_code, 200)

    # ---------------------------------------------------------------------- J
    def test_j_invalid_expired_and_redeemed_codes(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)

        invalid = self._redeem("A" * 16, user_csrf)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["detail"]["code"], "invalid_code")

        code = self._generate_code(admin_csrf)
        self.assertEqual(self._redeem(code, user_csrf).status_code, 200)
        used = self._redeem(code, user_csrf)
        self.assertEqual(used.status_code, 422)
        self.assertEqual(used.json()["detail"]["code"], "code_used")

        expired = "C" * 16
        self.database.execute(
            """
            INSERT INTO subscription_codes(code_hash, code_hint, duration_months, status, expires_at, created_at)
            VALUES (?, 'CCCC', 1, 'available', ?, ?)
            """,
            (token_hash(expired), (utc_now() - timedelta(days=1)).isoformat(), utc_now().isoformat()),
        )
        self.database.commit()
        expired_response = self._redeem(expired, user_csrf)
        self.assertEqual(expired_response.status_code, 422)
        self.assertEqual(expired_response.json()["detail"]["code"], "code_expired")
        self.assertTrue(self._user_status()["authenticated"])
        self.assertTrue(self._admin_status()["authenticated"])

    # ------------------------------------------------------------------ K / L
    def test_k_user_session_cannot_reach_admin_endpoints(self) -> None:
        user_csrf = self._user_register(self.browser)
        for path in ("/api/admin/dashboard", "/api/admin/users", "/api/admin/codes", "/api/admin/audit"):
            response = self.browser.get(path)
            self.assertEqual(response.status_code, 403, path)
            self.assertEqual(response.json()["detail"]["code"], "admin_forbidden")
        mutation = self.browser.post(
            "/api/admin/codes",
            headers={"X-CSRF-Token": user_csrf},
            json={"duration_months": 1, "quantity": 1},
        )
        self.assertEqual(mutation.status_code, 401)
        self.assertEqual(self._count("subscription_codes"), 0)
        self.assertFalse(self._admin_status()["authenticated"])

        # An admin account signed in through the *user* page does not get an
        # admin session either.
        other = self._new_browser()
        boot = other.get("/api/auth/csrf").json()["csrf_token"]
        login = other.post("/api/auth/login", headers={"X-CSRF-Token": boot}, json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        self.assertEqual(login.status_code, 200, login.text)
        self.assertEqual(other.get("/api/admin/dashboard").status_code, 403)
        self.assertFalse(self._admin_status(other)["authenticated"])

    def test_l_unauthenticated_redeem_is_rejected(self) -> None:
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)
        # Admin-only browser: there is no user session, so redeem must fail even
        # though an admin session and its CSRF token exist.
        response = self._redeem(code, admin_csrf)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["reason"], "no_user_session")
        anonymous = self._new_browser()
        response = self._redeem(code, "x" * 32, anonymous)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            self.database.execute("SELECT status FROM subscription_codes").fetchone()["status"],
            "available",
        )

    # ------------------------------------------------------------------ M / N
    def test_m_user_page_reload_keeps_user_session_and_csrf(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)
        reloaded_csrf = self._reload_user_page()
        self.assertEqual(reloaded_csrf, user_csrf)
        self.assertTrue(self._user_status()["authenticated"])
        # Reloading the admin page (new admin bootstrap) does not affect the user.
        self._reload_admin_page()
        self.assertEqual(self._redeem(code, reloaded_csrf).status_code, 200)

    def test_n_admin_page_reload_keeps_admin_session_and_csrf(self) -> None:
        self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        self._reload_user_page()
        reloaded = self._reload_admin_page()
        self.assertEqual(reloaded, admin_csrf)
        self.assertTrue(self._admin_status()["authenticated"])
        self._generate_code(reloaded)

    def test_user_page_bootstrap_does_not_break_pending_admin_login(self) -> None:
        admin_boot = self.browser.get("/api/admin/csrf").json()["csrf_token"]
        # User page opened in another tab refreshes its own bootstrap token.
        self.browser.get("/api/auth/csrf")
        response = self.browser.post(
            "/api/admin/login",
            headers={"X-CSRF-Token": admin_boot},
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.text)

    # ------------------------------------------------------------------ O / P
    def test_o_user_login_after_admin_logout(self) -> None:
        self._user_register(self.browser)
        user_csrf = self.browser.cookies.get("eh_csrf")
        self.assertEqual(self.browser.post("/api/auth/logout", headers={"X-CSRF-Token": user_csrf}).status_code, 200)
        admin_csrf = self._admin_login(self.browser)
        self.assertEqual(self.browser.post("/api/admin/logout", headers={"X-CSRF-Token": admin_csrf}).status_code, 200)
        new_user_csrf = self._user_login(self.browser)
        self.assertTrue(self._user_status()["authenticated"])
        self.assertFalse(self._admin_status()["authenticated"])
        admin_csrf = self._admin_login(self.browser)
        self.assertEqual(self._redeem(self._generate_code(admin_csrf), new_user_csrf).status_code, 200)

    def test_p_admin_login_after_user_logout(self) -> None:
        user_csrf = self._user_register(self.browser)
        self.assertEqual(self.browser.post("/api/auth/logout", headers={"X-CSRF-Token": user_csrf}).status_code, 200)
        admin_csrf = self._admin_login(self.browser)
        self.assertTrue(self._admin_status()["authenticated"])
        self.assertFalse(self._user_status()["authenticated"])
        self._generate_code(admin_csrf)

    # ---------------------------------------------------------------------- Q
    def test_q_user_page_and_admin_page_open_together(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        for _ in range(3):
            self.assertEqual(self.browser.get("/").status_code, 200)
            self.assertEqual(self.browser.get("/admin").status_code, 200)
            self.browser.get("/api/auth/csrf")
            self.browser.get("/api/admin/csrf")
            self.assertTrue(self._user_status()["authenticated"])
            self.assertTrue(self._admin_status()["authenticated"])
            self.assertEqual(self.browser.get("/api/admin/users").status_code, 200)
            self.assertEqual(self.browser.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)
        self.assertEqual(self.browser.cookies.get("eh_admin_csrf"), admin_csrf)

    # ---------------------------------------------------------------------- S
    def test_s_multiple_user_sessions_do_not_interfere(self) -> None:
        user_csrf = self._user_register(self.browser)
        second_browser = self._new_browser()
        second_csrf = self._user_register(second_browser, "second@example.com")
        admin_csrf = self._admin_login(self.browser)
        code = self._generate_code(admin_csrf)
        self.assertEqual(self._redeem(code, second_csrf, second_browser).status_code, 200)
        first = self.browser.get("/api/auth/me").json()["user"]
        second = second_browser.get("/api/auth/me").json()["user"]
        self.assertEqual(first["email"], USER_EMAIL)
        self.assertFalse(first["access"]["subscription_active"])
        self.assertTrue(second["access"]["subscription_active"])
        # Cross-browser CSRF use is rejected.
        other_code = self._generate_code(admin_csrf)
        self.assertEqual(self._redeem(other_code, second_csrf).json()["detail"]["code"], "csrf_invalid")

        # Two sessions of the same user (same device): logging one out keeps the other.
        second_tab = self._new_browser()
        device = next(cookie for cookie in self.browser.cookies.jar if cookie.name == "eh_device")
        second_tab.cookies.set("eh_device", device.value, domain=device.domain, path=device.path)
        tab_csrf = self._user_login(second_tab)
        self.assertEqual(self.browser.post("/api/auth/logout", headers={"X-CSRF-Token": user_csrf}).status_code, 200)
        self.assertTrue(self._user_status(second_tab)["authenticated"])
        self.assertEqual(self._redeem(other_code, tab_csrf, second_tab).status_code, 200)

    # ------------------------------------------------------------------ T / U
    def test_t_stale_admin_session_is_rejected_without_touching_user_session(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        self.database.execute(
            "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
            ((utc_now() - timedelta(minutes=1)).isoformat(), token_hash(self.browser.cookies.get("eh_admin_session"))),
        )
        self.database.commit()
        expired = self.browser.post("/api/admin/codes", headers={"X-CSRF-Token": admin_csrf}, json={"duration_months": 1, "quantity": 1})
        self.assertEqual(expired.status_code, 401)
        self.assertEqual(expired.json()["detail"]["code"], "session_expired")
        self.assertEqual(self._count("subscription_codes"), 0)
        self.assertTrue(self._user_status()["authenticated"])

        admin_csrf = self._admin_login(self.browser)
        self.database.execute("UPDATE users SET status = 'disabled' WHERE id = ?", (self.admin_id,))
        self.database.commit()
        self.assertEqual(self.browser.get("/api/admin/dashboard").status_code, 403)
        disabled = self.browser.post("/api/admin/codes", headers={"X-CSRF-Token": admin_csrf}, json={"duration_months": 1, "quantity": 1})
        self.assertEqual(disabled.status_code, 403)
        self.assertEqual(self._count("subscription_codes"), 0)
        self.assertFalse(self._admin_status()["authenticated"])
        self.assertTrue(self._user_status()["authenticated"])
        self.assertEqual(self.browser.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.browser.cookies.get("eh_csrf"), user_csrf)

    def test_u_stale_user_session_is_rejected_without_touching_admin_session(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        self.database.execute(
            "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
            ((utc_now() - timedelta(minutes=1)).isoformat(), token_hash(self.browser.cookies.get("eh_session"))),
        )
        self.database.commit()
        status = self._user_status()
        self.assertFalse(status["authenticated"])
        self.assertEqual(status["reason"], "session_expired")
        redeem = self._redeem("A" * 16, user_csrf)
        self.assertEqual(redeem.status_code, 401)
        self.assertTrue(self._admin_status()["authenticated"])
        self._generate_code(admin_csrf)

    def test_legacy_admin_session_in_user_cookie_is_neither_accepted_nor_deleted(self) -> None:
        """A pre-fix browser could hold an admin-scope token in eh_session."""
        admin_browser = self._new_browser()
        self._admin_login(admin_browser)
        admin_token = admin_browser.cookies.get("eh_admin_session")
        legacy = self._new_browser()
        legacy.cookies.set("eh_session", admin_token)
        status = self._user_status(legacy)
        self.assertFalse(status["authenticated"])
        self.assertEqual(legacy.get("/api/auth/me").json()["detail"]["reason"], "session_scope_mismatch")
        self.assertIsNotNone(self.database.execute("SELECT 1 FROM auth_sessions WHERE token_hash = ?", (token_hash(admin_token),)).fetchone())
        self.assertTrue(self._admin_status(admin_browser)["authenticated"])

    # ------------------------------------------------------------------ V / W
    def test_v_code_generation_and_audit_are_written_together(self) -> None:
        admin_csrf = self._admin_login(self.browser)
        before_audit = self._count("audit_logs")
        response = self.browser.post("/api/admin/codes", headers={"X-CSRF-Token": admin_csrf}, json={"duration_months": 3, "quantity": 3})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._count("subscription_codes"), 3)
        rows = self.database.execute(
            "SELECT actor_user_id, target_id FROM audit_logs WHERE action = 'generate_code' ORDER BY id"
        ).fetchall()
        self.assertEqual(self._count("audit_logs"), before_audit + 3)
        self.assertEqual({int(row["actor_user_id"]) for row in rows}, {self.admin_id})
        code_ids = {str(row["id"]) for row in self.database.execute("SELECT id FROM subscription_codes").fetchall()}
        self.assertEqual({row["target_id"] for row in rows}, code_ids)

    def test_w_audit_failure_leaves_no_partial_code(self) -> None:
        admin_csrf = self._admin_login(self.browser)
        before_codes = self._count("subscription_codes")
        before_audit = self._count("audit_logs")
        self.database.execute(
            """
            CREATE TEMP TRIGGER fail_generate_code_audit BEFORE INSERT ON audit_logs
            WHEN NEW.action = 'generate_code'
            BEGIN SELECT RAISE(ABORT, 'simulated audit failure'); END
            """
        )
        try:
            logging.disable(logging.CRITICAL)
            response = self.browser.post("/api/admin/codes", headers={"X-CSRF-Token": admin_csrf}, json={"duration_months": 1, "quantity": 2})
        finally:
            logging.disable(logging.NOTSET)
            self.database.execute("DROP TRIGGER IF EXISTS temp.fail_generate_code_audit")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "admin_code_generation_failed")
        self.assertEqual(self._count("subscription_codes"), before_codes)
        self.assertEqual(self._count("audit_logs"), before_audit)
        # The admin session and connection remain usable afterwards.
        self._generate_code(admin_csrf)

    def test_code_revoke_is_atomic_and_admin_scoped(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        self._generate_code(admin_csrf)
        code_id = int(self.database.execute("SELECT id FROM subscription_codes").fetchone()["id"])
        denied = self.browser.post(f"/api/admin/codes/{code_id}/revoke", headers={"X-CSRF-Token": user_csrf})
        self.assertEqual(denied.status_code, 403)
        revoked = self.browser.post(f"/api/admin/codes/{code_id}/revoke", headers={"X-CSRF-Token": admin_csrf})
        self.assertEqual(revoked.status_code, 200, revoked.text)
        audit = self.database.execute("SELECT actor_user_id FROM audit_logs WHERE action = 'revoke_code'").fetchall()
        self.assertEqual([int(row["actor_user_id"]) for row in audit], [self.admin_id])
        again = self.browser.post(f"/api/admin/codes/{code_id}/revoke", headers={"X-CSRF-Token": admin_csrf})
        self.assertEqual(again.status_code, 409)
        self.assertEqual(len(self.database.execute("SELECT 1 FROM audit_logs WHERE action = 'revoke_code'").fetchall()), 1)

    # ------------------------------------------------------------- diagnostics
    def test_auth_failures_are_logged_with_reason_and_without_secrets(self) -> None:
        user_csrf = self._user_register(self.browser)
        admin_csrf = self._admin_login(self.browser)
        with self.assertLogs("edge_hunter.auth", level="WARNING") as captured:
            self._redeem("A" * 16, admin_csrf)
            self._new_browser().post("/api/auth/redeem", headers={"X-CSRF-Token": user_csrf}, json={"code": "A" * 16})
        reasons = [getattr(record, "reason", None) for record in captured.records]
        codes = [getattr(record, "code", None) for record in captured.records]
        self.assertIn("csrf_invalid", codes)
        self.assertIn("no_user_session", reasons)
        secrets = (
            user_csrf,
            admin_csrf,
            self.browser.cookies.get("eh_session"),
            self.browser.cookies.get("eh_admin_session"),
        )
        for record in captured.records:
            rendered = repr(record.__dict__)
            for secret in secrets:
                self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()

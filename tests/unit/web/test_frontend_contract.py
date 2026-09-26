from __future__ import annotations

import re
import unittest
from pathlib import Path


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[3]
        cls.html = (root / "app" / "web" / "static" / "index.html").read_text(encoding="utf-8")
        cls.css = (root / "app" / "web" / "static" / "styles.css").read_text(encoding="utf-8")
        cls.js = (root / "app" / "web" / "static" / "app.js").read_text(encoding="utf-8")

    def test_rtl_branding_and_controls_exist(self) -> None:
        self.assertIn('dir="rtl"', self.html)
        self.assertIn("EDGE HUNTER", self.html)
        self.assertIn("Technical Market Analysis &amp; Signal Engine", self.html)
        for marker in ("symbol", "symbol-picker", "symbol-suggestions", "symbol-filters", "risk-percent", "capital", "lot-mode", "analyze-button", "login-form", "register-form", "redeem-form", "logout-button"):
            self.assertIn(f'id="{marker}"', self.html)

    def test_all_required_states_are_declared(self) -> None:
        for state in (
            "initial", "loading", "success", "weak", "medium", "strong", "very_strong",
            "no_clear_signal", "api_error", "data_unavailable", "session_expired", "unauthorized",
            "subscription_expired",
        ):
            self.assertRegex(self.js, rf"\b{re.escape(state)}\b")

    def test_public_auth_status_is_used_for_bootstrap(self) -> None:
        self.assertIn("/api/auth/status", self.js)
        admin_js = (Path(__file__).resolve().parents[3] / "app" / "web" / "static" / "admin.js").read_text(encoding="utf-8")
        # The admin page bootstraps from its own, admin-session-only status.
        self.assertIn("/api/admin/status", admin_js)
        self.assertNotIn("/api/auth/status", admin_js)
        # Protected /me remains available for server/API consumers, but public page
        # bootstrap must not trigger an expected 401 browser resource error.
        bootstrap_section = self.js.split("async function refreshMe()", 1)[1].split("async function submitAuth", 1)[0]
        self.assertNotIn("fetch('/api/auth/me')", bootstrap_section)

    def test_single_target_and_no_static_confidence_are_present(self) -> None:
        self.assertIn('id="target"', self.html)
        self.assertIn("id=\"stop-loss\"", self.html)
        self.assertNotIn("72%", self.html)
        self.assertNotIn("2,501.50", self.html)
        self.assertIn("/api/analyze", self.js)
        self.assertIn("/api/symbols/search", self.js)
        self.assertIn("setupSymbolPicker", self.js)
        self.assertIn("/api/auth/${kind}", self.js)
        self.assertIn("kind === 'login'", self.js)
        self.assertIn("X-CSRF-Token", self.js)

    def test_admin_dashboard_contract_exists(self) -> None:
        root = Path(__file__).resolve().parents[3]
        admin_html = (root / "app" / "web" / "static" / "admin.html").read_text(encoding="utf-8")
        admin_js = (root / "app" / "web" / "static" / "admin.js").read_text(encoding="utf-8")
        self.assertIn("لوحة الإدارة", admin_html)
        for route in ("/api/admin/login", "/api/admin/users", "/api/admin/codes", "/api/admin/audit"):
            self.assertIn(route, admin_js)

    def test_user_and_admin_pages_use_separate_session_tokens(self) -> None:
        admin_js = (Path(__file__).resolve().parents[3] / "app" / "web" / "static" / "admin.js").read_text(encoding="utf-8")
        # User page: session CSRF from its own cookie, never the admin one.
        self.assertIn("getCookie('eh_csrf')", self.js)
        self.assertNotIn("eh_admin_", self.js)
        self.assertNotIn("/api/admin/logout", self.js)
        # Admin page: admin cookie/endpoints only, never the user session ones.
        self.assertIn("cookie('eh_admin_csrf')", admin_js)
        self.assertIn("/api/admin/csrf", admin_js)
        self.assertIn("/api/admin/logout", admin_js)
        self.assertNotIn("'eh_csrf'", admin_js)
        self.assertNotIn("/api/auth/logout", admin_js)
        self.assertNotIn("/api/auth/csrf", admin_js)

    def test_responsive_css_and_no_horizontal_overflow(self) -> None:
        self.assertIn("overflow-x:auto", self.css.replace(" ", ""))
        self.assertIn("@media (max-width: 740px)", self.css)
        self.assertIn("grid-template-columns:1fr", self.css.replace(" ", ""))

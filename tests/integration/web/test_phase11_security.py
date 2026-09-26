from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.db.database import Database
from app.web.app import create_web_app
from config.config_hunter import load_settings


class Phase11SecurityIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        price = 2300.0
        with (root / "XAUUSD.csv").open("w", encoding="utf-8") as handle:
            handle.write("timestamp,open,high,low,close\n")
            for i in range(180):
                ts = start + timedelta(minutes=i)
                op = price
                cl = price + (0.2 if i % 2 else -0.1)
                hi = max(op, cl) + 0.4
                lo = min(op, cl) - 0.35
                handle.write(f"{ts.isoformat()},{op},{hi},{lo},{cl}\n")
                price = cl

        self.db = Database(root / "edge.db")
        settings = replace(
            load_settings(),
            database_path=root / "edge.db",
            cookie_secure=True,
            password_iterations=100_000,
            allowed_hosts=("testserver",),
            cors_origins=("https://allowed.example",),
            public_rate_limit=2,
            public_rate_limit_window_seconds=60,
            max_request_bytes=256,
            docs_enabled=False,
            environment="production",
            data_mode="local",
            live_provider_enabled=False,
            debug=False,
            trust_proxy_headers=False,
            log_dir=root / "logs",
        )
        self.app = create_web_app(root, database=self.db, settings=settings)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_security_headers_and_hsts(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "no-referrer")
        self.assertIn("max-age=31536000", response.headers.get("strict-transport-security", ""))
        self.assertTrue(response.headers.get("x-request-id"))

    def test_production_docs_are_disabled(self) -> None:
        self.assertEqual(self.client.get("/api/docs").status_code, 404)
        self.assertEqual(self.client.get("/api/openapi.json").status_code, 404)

    def test_explicit_cors_only(self) -> None:
        allowed = self.client.options(
            "/api/meta",
            headers={"Origin": "https://allowed.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "https://allowed.example")

        denied = self.client.options(
            "/api/meta",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertNotEqual(denied.headers.get("access-control-allow-origin"), "https://evil.example")

    def test_request_size_limit(self) -> None:
        body = b"x" * 1_500
        response = self.client.post("/api/auth/login", content=body)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "request_too_large")
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")

    def test_public_rate_limit(self) -> None:
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        self.assertEqual(self.client.get("/api/meta").status_code, 200)
        response = self.client.get("/api/meta")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "rate_limited")
        self.assertEqual(response.headers.get("retry-after"), "60")
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")

    def test_production_bootstrap_cookie_is_secure(self) -> None:
        response = self.client.get("/api/auth/csrf")
        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get("set-cookie", "").lower()
        self.assertIn("secure", cookie)

    def test_unhandled_errors_return_safe_json(self) -> None:
        async def explode() -> None:
            raise RuntimeError("private internal failure")

        self.app.add_api_route("/api/_test_error", explode, methods=["GET"])
        response = self.client.get("/api/_test_error")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "internal_error")
        self.assertNotIn("traceback", response.text.lower())
        self.assertNotIn("private internal failure", response.text)
        self.assertTrue(response.json()["detail"].get("request_id"))

    def test_readiness_does_not_leak_internal_details(self) -> None:
        original = self.db.execute
        self.db.execute = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("C:\\secret\\edge_hunter.db"))  # type: ignore[method-assign]
        try:
            response = self.client.get("/api/ready")
        finally:
            self.db.execute = original  # type: ignore[method-assign]
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text.lower())
        self.assertNotIn("edge_hunter.db", response.text.lower())

    def test_application_can_be_reinitialized_against_same_database(self) -> None:
        self.assertEqual(self.client.get("/api/ready").status_code, 200)
        settings = self.app.state.settings
        second = create_web_app(Path(self.tmp.name), database=self.db, settings=settings)
        second_client = TestClient(second)
        self.assertEqual(second_client.get("/api/health").status_code, 200)
        self.assertEqual(second_client.get("/api/ready").status_code, 200)

    def test_host_allowlist_rejects_unknown_host(self) -> None:
        response = self.client.get("/api/health", headers={"Host": "evil.example"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

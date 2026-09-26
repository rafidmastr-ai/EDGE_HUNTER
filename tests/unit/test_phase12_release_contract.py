from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "app" / "web" / "static"


class Phase12ReleaseContractTests(unittest.TestCase):
    def test_project_manifests_declare_httpx2(self) -> None:
        requirements = (PROJECT_ROOT / "requirements-web.txt").read_text(encoding="utf-8")
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^httpx2>=2\.0,<3\.0\s*$")
        self.assertIn('"httpx2>=2.0,<3.0"', pyproject)

    def test_environment_example_contains_no_raw_credentials(self) -> None:
        env_text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotRegex(env_text, r"(?im)^\s*(password|admin_password|secret|token)\s*=")
        self.assertIn("EDGE_HUNTER_ALLOWED_HOSTS", env_text)
        self.assertIn("EDGE_HUNTER_COOKIE_SECURE", env_text)

    def test_required_runtime_files_exist(self) -> None:
        required = [
            PROJECT_ROOT / "main.py",
            PROJECT_ROOT / "config" / "config_hunter.py",
            PROJECT_ROOT / "app" / "db" / "migrations.py",
            PROJECT_ROOT / "app" / "data" / "pipeline.py",
            PROJECT_ROOT / "app" / "features" / "engine.py",
            PROJECT_ROOT / "app" / "strategies" / "registry.py",
            PROJECT_ROOT / "app" / "backtest" / "engine.py",
            PROJECT_ROOT / "app" / "research" / "runner.py",
            PROJECT_ROOT / "app" / "optimization" / "runner.py",
            PROJECT_ROOT / "app" / "signals" / "engine.py",
            PROJECT_ROOT / "app" / "web" / "app.py",
            PROJECT_ROOT / "app" / "auth" / "service.py",
            PROJECT_ROOT / "app" / "admin" / "service.py",
            PROJECT_ROOT / "app" / "web" / "security.py",
            PROJECT_ROOT / "scripts" / "backup_database.py",
            PROJECT_ROOT / "scripts" / "restore_database.py",
            PROJECT_ROOT / "scripts" / "run_production.ps1",
        ]
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
        self.assertEqual(missing, [])

    def test_frontend_contract_is_real_dynamic_and_rtl(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        js = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        css = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn('<html lang="ar" dir="rtl">', html)
        self.assertIn('name="viewport"', html)
        self.assertIn('/api/analyze', js)
        self.assertIn("navigator.clipboard", js)
        self.assertIn("@media (max-width: 740px)", css)
        self.assertIn("@media (max-width: 420px)", css)
        self.assertIn("overflow-x: hidden", css)
        self.assertNotIn("background-image: url(", css.lower())

    def test_no_hardcoded_market_price_or_confidence_in_frontend(self) -> None:
        combined = "\n".join(
            (STATIC_ROOT / name).read_text(encoding="utf-8")
            for name in ("index.html", "app.js", "styles.css")
        )
        suspicious_price = re.compile(r"\b(?:1\d{3}|2\d{3}|3\d{3}|4\d{3})\.\d{1,4}\b")
        suspicious_confidence = re.compile(r"(?:confidence\s*[:=]\s*|ثقة\s*[:=]\s*)\d{1,3}(?:\.\d+)?")
        self.assertIsNone(suspicious_price.search(combined))
        self.assertIsNone(suspicious_confidence.search(combined))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "app" / "web" / "static"


class Phase12VisualQATests(unittest.TestCase):
    def test_desktop_tablet_mobile_breakpoints_and_no_overflow_contract(self) -> None:
        css = (STATIC / "styles.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"@media\s*\(max-width:\s*1100px\)")
        self.assertRegex(css, r"@media\s*\(max-width:\s*740px\)")
        self.assertRegex(css, r"@media\s*\(max-width:\s*420px\)")
        self.assertIn("overflow-x: hidden", css)
        self.assertIn("overflow:auto", css)

    def test_rtl_viewports_and_accessibility_basics(self) -> None:
        for filename in ("index.html", "admin.html"):
            text = (STATIC / filename).read_text(encoding="utf-8")
            self.assertIn('dir="rtl"', text)
            self.assertIn('name="viewport"', text)
            self.assertIsNotNone(re.search(r'charset="utf-8"', text, flags=re.IGNORECASE))

    def test_required_user_states_and_copy_controls_are_present(self) -> None:
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        required_states = {
            "initial", "loading", "success", "weak", "medium", "strong",
            "very_strong", "no_clear_signal", "api_error", "data_unavailable",
            "session_expired", "unauthorized", "subscription_expired",
        }
        found = set(re.findall(r"^\s*([a-z_]+):\s*\{", js, flags=re.MULTILINE))
        self.assertTrue(required_states.issubset(found))
        for value in ("entry", "target", "stop-loss"):
            self.assertIn(f'data-copy="{value}"', html)
        self.assertIn("navigator.clipboard", js)

    def test_frontend_has_no_embedded_reference_screenshot(self) -> None:
        for filename in ("index.html", "admin.html", "app.js", "admin.js", "styles.css"):
            text = (STATIC / filename).read_text(encoding="utf-8").lower()
            self.assertNotIn("base64,image", text)
            self.assertNotRegex(text, r"url\([^)]+\.(?:png|jpg|jpeg|webp)")

    def test_frontend_javascript_syntax_when_node_is_available(self) -> None:
        node = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if node.returncode != 0:
            self.skipTest("Node.js is not installed; browser syntax check can be run via scripts/run_visual_qa.py")
        for filename in ("app.js", "admin.js"):
            result = subprocess.run(
                ["node", "--check", str(STATIC / filename)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

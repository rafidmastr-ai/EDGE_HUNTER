"""Browser-based responsive/visual QA for Phase 12.

Requires: playwright Python package and a usable Chromium executable.
The script intentionally checks behavior/geometry rather than comparing pixels
against a static screenshot, because the Phase 09 reference image is visual guidance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8000/"
VIEWPORTS = {
    "desktop": (1440, 1000),
    "tablet": (1024, 900),
    "mobile": (390, 844),
}


def discover_chromium() -> str | None:
    for candidate in ("chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def wait_for_server(url: str, process: subprocess.Popen[str] | None, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "unknown error"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            output = ""
            if process.stdout:
                try:
                    output = process.stdout.read()[-4000:]
                except Exception:
                    pass
            raise RuntimeError(f"Web server exited before becoming ready (code={process.returncode}).\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"Web server did not become ready within {timeout:.0f}s: {url} ({last_error})")


def start_server(url: str) -> tuple[subprocess.Popen[str] | None, str]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    env = dict(__import__("os").environ)
    env.update(
        {
            "EDGE_HUNTER_ENV": "development",
            "EDGE_HUNTER_ALLOWED_HOSTS": f"{host},127.0.0.1,localhost",
            "EDGE_HUNTER_API_HOST": host,
            "EDGE_HUNTER_API_PORT": str(port),
            "EDGE_HUNTER_COOKIE_SECURE": "false",
            "EDGE_HUNTER_DOCS_ENABLED": "true",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.web.app:app", "--host", host, "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, url


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER browser visual QA")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "visual_qa")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        python_exe = Path(sys.executable)
        print("VISUAL QA BLOCKED: Playwright is unavailable.")
        print(f"Python used: {python_exe}")
        print(f"Install dependencies: \"{python_exe}\" -m pip install -r requirements-web.txt")
        print(f"Install Chromium: \"{python_exe}\" -m playwright install chromium")
        print(f"Original error: {exc}")
        return 2

    browser_exe = discover_chromium()
    process = None
    if not args.no_start_server:
        process, _ = start_server(args.url)
        wait_for_server(args.url, process)
    results: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            launch_kwargs = {"headless": True}
            if browser_exe:
                launch_kwargs["executable_path"] = browser_exe
            browser = playwright.chromium.launch(**launch_kwargs)
            for name, (width, height) in VIEWPORTS.items():
                page = browser.new_page(viewport={"width": width, "height": height})
                console_errors: list[str] = []
                page_errors: list[str] = []
                page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                # Do not require network-idle: the application may keep background
                # connections/polling alive. DOMContentLoaded is the stable visual-QA
                # readiness point; a short settle period lets styles/scripts render.
                response = page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(750)
                body_width = page.evaluate("document.body.scrollWidth")
                client_width = page.evaluate("document.documentElement.clientWidth")
                result = {
                    "viewport": name,
                    "width": width,
                    "height": height,
                    "status": response.status if response else None,
                    "rtl": page.locator("html").get_attribute("dir") == "rtl",
                    "horizontal_overflow": body_width > client_width + 1,
                    "console_errors": console_errors,
                    "page_errors": page_errors,
                }
                page.screenshot(path=str(args.output_dir / f"{name}.png"), full_page=True)
                results.append(result)
                page.close()
            browser.close()
    except Exception as exc:
        print(f"VISUAL QA ERROR: {exc}")
        return 1
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    report = {
        "viewports": results,
        "passed": all(
            item["status"] == 200
            and item["rtl"]
            and not item["horizontal_overflow"]
            and not item["console_errors"]
            and not item["page_errors"]
            for item in results
        ),
    }
    (args.output_dir / "visual_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

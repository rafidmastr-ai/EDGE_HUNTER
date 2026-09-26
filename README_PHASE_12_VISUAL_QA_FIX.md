# Phase 12 — Visual QA Dependency Fix

The browser-based Phase 12 visual QA was blocked because the active project `.venv` did not contain the Python `playwright` package.

This patch updates the dependency manifests and adds a Windows setup script that uses the project's own `.venv` Python to install Playwright and its Chromium browser.

## Files in this patch

- `requirements-web.txt`
- `pyproject.toml`
- `scripts/run_visual_qa.py`
- `scripts/setup_visual_qa.ps1`

## Windows steps

From the project root:

```powershell
.\scripts\setup_visual_qa.ps1
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\run_visual_qa.py
```

The runner checks desktop/tablet/mobile viewports, HTTP 200, RTL, horizontal overflow, browser console errors, page errors, and writes screenshots plus `reports/visual_qa/visual_qa.json`.

Playwright's official Python documentation requires installing the Python package and then installing the browser binaries; for Chromium, `python -m playwright install chromium` is the targeted setup command.

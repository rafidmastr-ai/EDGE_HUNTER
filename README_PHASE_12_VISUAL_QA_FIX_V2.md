# Phase 12 Visual QA Fix V2

The previous Visual QA run successfully installed Playwright/Chromium but timed out waiting for `networkidle` while navigating to `http://127.0.0.1:8000/`.

This patch changes only `scripts/run_visual_qa.py`:

- starts the local Uvicorn server when needed;
- waits for the HTTP endpoint to become ready instead of using a fixed 2-second sleep;
- fails with the server's captured output if Uvicorn exits early;
- uses `domcontentloaded` plus a short settle delay instead of `networkidle`, because a dynamic web app can legitimately keep network activity alive;
- keeps the existing desktop/tablet/mobile, RTL, overflow, console-error, page-error and screenshot checks.

Run:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_visual_qa.py
```

Or, if the web server is already running:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_visual_qa.py --no-start-server
```

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "VISUAL QA SETUP FAILED: .venv Python not found at $Python" -ForegroundColor Red
    exit 1
}

Write-Host "Installing/updating Phase 12 web test dependencies..."
& $Python -m pip install -r (Join-Path $Root "requirements-web.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing Playwright Chromium browser..."
& $Python -m playwright install chromium
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "VISUAL QA SETUP PASS"
Write-Host "Run next:"
Write-Host "  & `"$Python`" scripts\run_visual_qa.py"

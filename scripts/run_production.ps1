$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "EDGE HUNTER .venv Python was not found: $Python"
}

if (-not $env:EDGE_HUNTER_ENV) { $env:EDGE_HUNTER_ENV = "production" }
if ($env:EDGE_HUNTER_ENV -ne "production") { throw "EDGE_HUNTER_ENV must be production for this script." }
if (-not $env:EDGE_HUNTER_ALLOWED_HOSTS) { throw "EDGE_HUNTER_ALLOWED_HOSTS must be configured before public launch." }
if (-not $env:EDGE_HUNTER_DATA_MODE) { $env:EDGE_HUNTER_DATA_MODE = "live" }
if ($env:EDGE_HUNTER_DATA_MODE -ne "live") { throw "Production Web UI must use EDGE_HUNTER_DATA_MODE=live." }
if (-not $env:EDGE_HUNTER_LIVE_PROVIDER) { $env:EDGE_HUNTER_LIVE_PROVIDER = "twelvedata" }
if ($env:EDGE_HUNTER_LIVE_PROVIDER -ne "twelvedata") { throw "EDGE HUNTER production Web UI expects the configured Twelve Data provider." }
if (-not $env:EDGE_HUNTER_LIVE_PROVIDER_ENABLED) { $env:EDGE_HUNTER_LIVE_PROVIDER_ENABLED = "true" }
if ($env:EDGE_HUNTER_LIVE_PROVIDER_ENABLED -notin @("true", "1", "yes", "on")) { throw "EDGE_HUNTER_LIVE_PROVIDER_ENABLED must be true in production." }
if (-not $env:EDGE_HUNTER_API_HOST) { $env:EDGE_HUNTER_API_HOST = "127.0.0.1" }
if (-not $env:EDGE_HUNTER_API_PORT) { $env:EDGE_HUNTER_API_PORT = "8000" }
$env:EDGE_HUNTER_DEBUG = "false"
$env:EDGE_HUNTER_COOKIE_SECURE = "true"
$env:EDGE_HUNTER_DOCS_ENABLED = "false"

Write-Host "EDGE HUNTER - production preflight" -ForegroundColor Cyan
& $Python main.py --serve --production
exit $LASTEXITCODE

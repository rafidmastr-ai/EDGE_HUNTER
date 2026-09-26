# EDGE HUNTER — PHASE 12
## Final Integration, Regression & Visual QA

Phase 12 is the release gate before the live-data provider phase. It does not redesign the core architecture. It integrates and regression-tests the layers already built in Phases 01–11.

## Automated coverage

The default test runner discovers the entire `tests/` tree. The Phase 12 additions add:

- clean-install/compile and dependency-manifest checks;
- migration and public-database exposure regression checks;
- one complete user journey: register → trial → analyze → trial expiry → subscription code → restored access → analyze;
- admin code generation and dashboard verification;
- frontend RTL/responsive/copy-control/static-content checks;
- Node JavaScript syntax checks when Node.js is installed.

The existing Phase 01–11 test suites remain part of the same run.

## Visual QA

`scripts/run_visual_qa.py` performs browser checks at 1440×1000, 1024×900 and 390×844. It verifies HTTP 200, RTL, no horizontal overflow, no browser console errors and no page errors, and saves screenshots under `reports/visual_qa/`.

The browser check is deliberately not a pixel-diff test. The Phase 09 reference image is a visual direction, while the production page is a real API-driven interface rather than a screenshot background.

Run:

```powershell
python scripts/run_visual_qa.py
```

When Chromium/Playwright is unavailable, the script exits with a clear blocked status rather than pretending visual QA passed.

## Release procedure

Run the complete release check:

```powershell
python scripts/run_phase12_release.py
```

Or use the normal application entry point, which already executes the complete discovered test suite before startup:

```powershell
python main.py
```

The release script writes:

- `reports/phase12_release_report.json`
- `reports/phase12_release_readiness.md`

## End-to-end acceptance

The integrated flow proves server-side subscription enforcement. The UI may display weak/no-clear states, but no protected analysis is allowed after trial/subscription expiry until access is restored.

## Known limitations carried forward

1. Browser pixel comparison against the supplied visual reference is not automated; the browser script checks runtime behavior and geometry instead.
2. The Phase 09 historical UI symbol contract includes EURUSD, while the currently available historical files contain XAUUSD and GBPUSD. The system returns a controlled data-unavailable state for unavailable symbols; the live provider phase is the appropriate place to expand the runtime market universe.
3. Public deployment still depends on an external HTTPS edge/tunnel or reverse proxy as documented in Phase 11.

## Release gate

The automated Phase 12 gate is considered passed when the full test suite and compile check pass. Final release readiness remains pending until browser QA has been executed in a real Chromium environment without errors/overflow.

Stop here before adding or activating live market data logic.

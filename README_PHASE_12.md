# EDGE HUNTER — Phase 12

Final integration, regression and visual-QA gate before Phase 13 live market data.

## What was added

- full-system end-to-end regression;
- release/installation/configuration contract checks;
- final RTL/responsive/frontend static QA checks;
- browser-based visual QA runner for desktop/tablet/mobile;
- machine-readable and Markdown release reports;
- final `main.py` startup message for Phase 12.

## Run

```powershell
python scripts/run_phase12_release.py
```

Then, on a machine with Chromium/Playwright:

```powershell
python scripts/run_visual_qa.py
```

The normal entry point continues to run the complete test suite before application startup:

```powershell
python main.py
```

Phase 12 must stop here. Do not add live-data behavior until this gate is verified.

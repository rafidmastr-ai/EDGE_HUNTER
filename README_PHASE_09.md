# Phase 09 — EDGE HUNTER Web UI

Implemented the frontend/API boundary described by the Phase 09 specification.

## New modules

- `app/web/app.py` — FastAPI application and API routes.
- `app/web/schemas.py` — request/response contracts.
- `app/web/analysis_service.py` — local OHLC loading, MTF resampling and calls into the existing feature/strategy/confidence stack.
- `app/web/static/index.html` — RTL dashboard.
- `app/web/static/styles.css` — responsive dark financial UI.
- `app/web/static/app.js` — API integration, state rendering, copying and real candle chart.

## Test coverage

Phase 09 tests are under `tests/unit/web/` and `tests/integration/web/`.

The `main.py` project runner discovers them together with earlier phase tests.

## Start web server

From the project root:

```powershell
python -m uvicorn app.web.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/`.

The API does not expose the SQLite database directly. The UI uses the existing domain/analysis layers and returns structured JSON.

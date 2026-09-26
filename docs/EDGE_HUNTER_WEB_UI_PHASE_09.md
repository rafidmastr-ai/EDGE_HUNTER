# EDGE HUNTER — Phase 09 Web UI

## Scope

Phase 09 adds the responsive RTL web frontend and a thin API layer over the already-built Feature / Strategy / Signal Confidence modules.

## UI contract

- Arabic RTL premium dark dashboard.
- EDGE HUNTER branding and subtitle.
- Symbol selector: EURUSD / XAUUSD.
- Risk presets plus Custom.
- Optional Capital input.
- Auto / Manual lot mode.
- Strategy selection remains internal to the analysis engine.
- One final target only.
- Capital-dependent profit/loss values are hidden when Capital is absent.
- Chart is drawn from API candle data; no screenshot/background is used.

## API

`GET /api/health` checks service reachability.

`GET /api/meta` returns supported UI metadata.

`POST /api/analyze` consumes the analysis form and returns the structured Phase 08 result plus:

- selected multi-timeframe context,
- chart candles,
- strategy/timeframe comparison,
- optional capital impact,
- source metadata.

The local API reads historical CSV data from `data/raw/` and supports epoch-millisecond timestamps as well as ISO timestamps because the available datasets use both forms.

## Timeframe policy

The user does not select a timeframe. The service evaluates M5, M15 and H1 and applies a deterministic multi-timeframe agreement rule. The resulting selected context is returned as `primary_timeframe`.

This is a presentation/runtime policy, not a claim that the chosen timeframe is optimal.

## Error/state contract

The browser includes explicit rendering for initial, loading, success, weak, medium, strong, very strong, no-clear-signal, API error, data unavailable, session expired, unauthorized and subscription expired.

The latter three are prepared for the authentication/subscription phase; Phase 09 itself does not create authentication or entitlement state.

## Running locally

The web app can be started with Uvicorn using the `app.web.app:app` target. The regular `main.py` remains a non-blocking automated test/startup check; it does not keep a web server process open by default.

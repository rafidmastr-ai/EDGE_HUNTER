# Phase 13 — Provider Review / Implementation Status

## Selected adapter

Twelve Data is implemented as the first concrete adapter behind the provider-neutral EDGE HUNTER market-data boundary.

## Verified provider facts

The current Twelve Data documentation identifies:

- REST API base: `https://api.twelvedata.com`.
- `/time_series` with intraday intervals including 1min, 5min, 15min, 30min, 1h and 4h.
- `XAU/USD` as a gold spot instrument in commodity reference data.
- API-key authentication through either query parameters or HTTP header; the header form is recommended.
- API credit quotas; the current Basic individual plan shows 8 API credits/minute and 800/day.
- A 5,000-point maximum per time-series request.
- Individual-plan usage restrictions for personal/internal/non-commercial use; commercial display/redistribution requires the appropriate business arrangement and exchange licensing where applicable.

See the Twelve Data documentation and support pages referenced in `docs/LIVE_DATA_PROVIDER_PHASE_13.md` before production enablement.

## Implementation status

- Provider boundary preserved.
- Twelve Data response mapped into canonical `OHLCBar`.
- UTC timestamp normalization.
- Symbol mapping `EURUSD -> EUR/USD` and `XAUUSD -> XAU/USD`.
- Input validation, duplicate collapse and candle-geometry validation.
- Bounded retry handling for transient network errors, 429 and 5xx.
- 401/403 are not retried.
- Local rate limiting and short-lived cache.
- Safe health endpoint and controlled live-unavailable error.
- Production live mode fails closed when provider is unconfigured.
- No provider logic added to strategies or backtest code.
- Mocked provider response tests and live-path integration tests are included.

## Final gate before public use

The project can be switched to live mode only after the owner supplies a valid provider key and confirms the provider plan/licensing is suitable for the intended public display/paid service. Run `scripts/check_live_provider.py` as the explicit network smoke test, then validate the resulting live analysis against a controlled known dataset before enabling public access.

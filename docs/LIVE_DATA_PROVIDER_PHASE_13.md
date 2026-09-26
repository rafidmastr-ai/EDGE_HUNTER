# EDGE HUNTER — Phase 13 Live Data Provider Adapter

## Scope

Phase 13 connects a real provider behind the existing market-data boundary. The analysis, strategy and backtest interfaces remain provider-neutral.

## Provider implementation

The first concrete adapter is **Twelve Data**. The provider is configurable and can be replaced later without changing strategies or the backtest engine.

### Current provider facts verified from Twelve Data documentation

- REST base URL: `https://api.twelvedata.com`.
- The `/time_series` endpoint supports intraday intervals including `1min`, `5min`, `15min`, `30min`, `1h`, and `4h`.
- `XAU/USD` is listed as a gold spot instrument in the provider's commodity reference data.
- Twelve Data documents both query-parameter and HTTP-header authentication and recommends the header form `Authorization: apikey <key>`.
- API credits are quota-based. The current Basic individual plan shows 8 API credits/minute and 800/day; the adapter therefore has a conservative local 8-request/minute default and configurable limits.
- A single `time_series` request is documented as supporting up to 5,000 data points; EDGE HUNTER caps the configured fetch at 5,000 and defaults to 800 bars per timeframe.
- Twelve Data documents retry handling for `429` responses and caching as a best practice; the adapter implements bounded retries and short-lived in-memory caching.

Provider documentation and commercial-use terms must be checked again before public production deployment because plans and licensing can change.

## Licensing / public deployment note

The current Twelve Data individual plans are intended for personal/internal and non-commercial usage. Their support guidance states that commercial display and redistribution require business arrangements and exchange licensing as applicable. Therefore, do **not** enable Twelve Data public/commercial display using an individual plan merely because the API responds successfully.

## Configuration

Use runtime environment variables; never place a real key in source control.

```text
EDGE_HUNTER_DATA_MODE=live
EDGE_HUNTER_LIVE_PROVIDER=twelvedata
EDGE_HUNTER_LIVE_PROVIDER_ENABLED=true
EDGE_HUNTER_LIVE_PROVIDER_API_KEY=<secret>
EDGE_HUNTER_LIVE_PROVIDER_URL=https://api.twelvedata.com
EDGE_HUNTER_LIVE_PROVIDER_TIMEOUT_SECONDS=10
EDGE_HUNTER_LIVE_PROVIDER_MAX_RETRIES=2
EDGE_HUNTER_LIVE_PROVIDER_BACKOFF_SECONDS=0.5
EDGE_HUNTER_LIVE_PROVIDER_CACHE_TTL_SECONDS=15
EDGE_HUNTER_LIVE_PROVIDER_MAX_BARS=800
EDGE_HUNTER_LIVE_PROVIDER_RATE_LIMIT_PER_MINUTE=8
```

Compact symbols are translated internally to Twelve Data's `BASE/QUOTE` form for Forex, Metals and Crypto (`XAUUSD` → `XAU/USD`, `GBPUSD` → `GBP/USD`, `BTCUSD` → `BTC/USD`, `ETHUSDT` → `ETH/USDT`); symbols already in `BASE/QUOTE` form (as returned by the symbol catalog) are sent unchanged. Provider-specific symbols never enter the strategy layer.

## Runtime behavior

- Live requests are converted directly to canonical `OHLCBar` objects.
- Timestamps are normalized to UTC.
- Duplicate timestamps are collapsed deterministically.
- Invalid/non-positive OHLC rows are ignored.
- Invalid candle geometry is rejected.
- Provider credentials are sent through the recommended HTTP authorization header and are never returned by health/meta endpoints.
- Retries are limited to transient network failures, `429`, and `5xx` responses.
- Provider `401/403` responses are not retried.
- Short-lived in-memory caching reduces repeated requests.
- A controlled `503` is returned when live data is required and unavailable.
- Live analysis (`EDGE_HUNTER_DATA_MODE=live`, the default) never falls back to local CSV, in any environment. Local CSV files are for backtesting/learning only; `EDGE_HUNTER_DATA_MODE=local` is an explicit offline mode. The former `EDGE_HUNTER_LIVE_FALLBACK_TO_LOCAL` setting was removed.

## Health

`GET /api/live-health` returns safe provider status only:

- provider name
- configured flag
- health state
- last success/failure time
- request count
- cache hits
- latency
- last internal error code

No API key is returned.

## Replacing the provider later

Implement the same `LiveMarketDataProvider` boundary, add the provider to `app/providers/factory.py`, and map its payload into canonical `OHLCBar`. Strategies, feature calculations, signal scoring and backtests must not be modified solely to support the new provider.

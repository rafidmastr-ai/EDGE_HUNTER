# EDGE HUNTER — Architecture Phase 01

## Dependency direction

```text
main.py
   |
   v
app/core/bootstrap.py
   |
   +--> config/config_hunter.py
   +--> app/db/*
   +--> app/providers/*
   +--> app/domain/*
```

The domain layer does not depend on API/UI, database implementation, or a specific market-data vendor.

## Provider boundary

`MarketDataProvider` defines the canonical boundary.

- `CsvHistoricalDataProvider`: reserved for Phase 02 implementation.
- `LiveMarketDataProvider`: reserved for Phase 13 implementation.

Strategies and analysis must consume canonical domain data, not provider-specific objects.

## Database boundary

Application code uses `Database` rather than constructing SQLite connections throughout the project.

`MigrationRunner` applies versioned migrations in deterministic order.

## Configuration

`config/config_hunter.py` is the central configuration interface.

Sensitive values are loaded from environment variables. No password or API secret is committed in this phase.

## Deliberate non-goals

Phase 01 does not implement:

- strategy rules
- indicators/features
- backtesting
- optimization
- confidence scoring
- web UI
- authentication/subscriptions
- public deployment
- live market-data provider integration

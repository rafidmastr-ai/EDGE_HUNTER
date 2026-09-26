# Phase 05 — Backtest Engine

Implemented a deterministic candle/event-based backtest engine on top of the Phase 01–04 contracts.

## New package

`app/backtest/`

- `config.py` — execution, costs, intrabar policy and sizing.
- `models.py` — trade, metrics and result contracts.
- `engine.py` — event/candle execution engine.
- `metrics.py` — performance/stability metrics.
- `report.py` — JSON serialization/export.
- `__init__.py` — public package exports.

## Tests

Phase 05 tests live under `tests/unit/backtest/` and `tests/integration/backtest/` and are included in the project's existing automated `main.py` test runner.

The tests use synthetic OHLC data with known outcomes and cover lifecycle timing, look-ahead prevention, intrabar ambiguity, costs, sizing, expiry, metrics and report serialization.

## Boundary

No strategy optimization is performed in Phase 05. Research and optimization remain in the later dedicated phases.

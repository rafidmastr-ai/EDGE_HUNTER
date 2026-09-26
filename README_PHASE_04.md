# README — Phase 04

Phase 04 adds the modular Strategy Engine to EDGE HUNTER.

Implemented strategy families:

- `ClassicV1`
- `SMCV1`
- `ICTV1`

Use `StrategyRegistry.default()` to run all initial families through the same interface.

## Validation

Run the complete suite through `main.py`, which executes `unittest` discovery before application startup. The Phase 04 build was validated with 44 total tests across Phases 01–04.

```powershell
python main.py
```

The dedicated Phase 04 tests cover contracts, timing, no-look-ahead behavior, dynamic R:R bounds, one-target output, edge cases, and registry integration.

Phase 04 is a framework and baseline strategy implementation. Backtest profitability and comparative superiority are deliberately deferred to Phase 05/06/07.

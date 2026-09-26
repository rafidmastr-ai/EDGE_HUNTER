# EDGE HUNTER — Phase 05 Backtest Engine

## Purpose

Phase 05 provides a reusable candle/event backtesting instrument. It consumes canonical OHLC data and a Phase 04 strategy through the common `Strategy` contract. This phase measures strategy behavior; it does not optimize or declare a strategy superior.

## Execution model

- A strategy is evaluated at each canonical bar timestamp `T` with `StrategyContext` containing only bars/snapshots through `T`.
- Phase 04's initial strategies define entry at the decision-bar close. The engine records the entry at that close.
- TP/SL monitoring starts on the **next** candle. The signal candle itself is never used to determine whether TP or SL was touched.
- Only one position is active at a time in this phase. Signals produced while an active trade exists are ignored.
- When the dataset ends with an open trade, the engine records an explicit `DATA_END` expiry rather than silently dropping it.

## Intrabar ambiguity

With OHLC data, a candle can touch both the target and stop without revealing the true tick/order sequence. The engine therefore requires an explicit policy:

- `STOP_FIRST` (default): conservative deterministic classification as a loss.
- `TARGET_FIRST`: deterministic classification as a win.
- `SKIP_TRADE`: record an `AMBIGUOUS_SKIPPED` event, excluded from executed-trade metrics.

This is a modeling rule, not a claim about what happened inside the candle.

## Costs

`spread` is the full quoted spread and is applied adversely as half on entry and half on exit. `commission_per_unit` is charged once at entry and once at exit. Trigger comparisons use the raw OHLC levels; this keeps the execution-cost model explicit rather than pretending OHLC contains bid/ask tick detail.

## Position sizing

The engine supports:

- normalized fixed units (`default_position_units`);
- manual lot sizing (`lot_size * contract_size`);
- risk-based sizing (`risk_per_trade_pct`) using current equity when capital is available.

A risk-sized trade is based on the absolute entry-to-stop distance. Later research can decide whether these assumptions match each instrument's contract specification.

## Metrics

The engine produces total executed trades, wins, losses, expiries, ambiguity count, win rate, profit factor, expectancy, average planned R:R, max drawdown, average trade, net return when starting capital is supplied, TP/SL/expiry rates, return dispersion, period win-rate dispersion and positive-period ratio.

Profit factor is `None` when no losing trade exists because the mathematical denominator is zero rather than a finite number.

## Reproducibility

Each result contains metadata including engine version, strategy identity, analysis version, bar count, a SHA-256 fingerprint of the canonical OHLC input, execution configuration and explicit timing/look-ahead policies. `generated_at_utc` is operational metadata and is intentionally excluded from the stable reproducibility metadata.

## Export

Results can be converted through `BacktestResult.to_dict()` or `app.backtest.report.to_json()` and saved using `save_json()`.

## Research boundary

Optimization, parameter search, train/validation/OOS selection, walk-forward analysis and strategy comparison belong to later phases. Phase 05 is the measurement instrument only.

# EDGE HUNTER — Phase 06: Backtest Research & Strategy Comparison

## Purpose

Phase 06 is the dedicated research layer above the Phase 05 backtest engine. It compares the initial Classic, SMC and ICT families across symbols, timeframes and configurable parameter variants, keeping in-sample (IS) and out-of-sample (OOS) observations separate.

## Research Matrix

The default matrix contains:

- Symbols: `XAUUSD`, `GBPUSD`
- Timeframes: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`
- Strategy families: `Classic`, `SMC`, `ICT`
- R:R research coverage: fixed `1.50`, fixed `1.75`, fixed `2.00`, plus the existing dynamic `1.50–2.00` range
- Additional existing Phase 04 parameter variants for candle-body confirmation and structure lookback

These are experiment configurations, not claims that any one setting is optimal.

## IS / OOS Handling

The default split is chronological 70% IS / 30% OOS. OOS backtests retain the full pre-OOS history for feature warm-up and strategy context, while a period gate prevents any trade from being created before the OOS start timestamp. This avoids resetting indicator history at the split boundary.

## Metrics

Every experiment records, consistently for IS and OOS:

- Trade count, wins, losses, expiry and ambiguity
- Win rate
- Profit factor
- Expectancy in price/PnL units
- Mean realized R multiple
- Average planned R:R
- Maximum drawdown and drawdown percentage when starting capital exists
- Net return when starting capital exists
- TP/SL/expiry rates
- Return dispersion and chronological period consistency
- Signal frequency and signal-to-trade ratio

## Stability / Failure Analysis

The research layer adds evidence flags such as:

- `INSUFFICIENT_SAMPLE`
- `OVERTRADED`
- `UNSTABLE`
- `DATA_SENSITIVE`
- `PROMISING` (only when all configured Phase 07 screening gates pass)
- `NOT_CANDIDATE`

A configuration may carry multiple evidence flags. No global ranking is produced.

## Regime / Period Context

The Phase 05 metrics already include chronological stability buckets. Research results preserve those measures and keep the exact IS/OOS split. The framework is intentionally ready for additional regime segmentation without changing the strategy interface.

## Reproducibility

Every run records:

- Research, feature and backtest engine versions
- Dataset hashes
- Source paths when supplied
- Exact matrix and strategy parameters
- Backtest execution configuration
- IS/OOS boundaries
- Experiment IDs

## Machine-Readable Outputs

`save_research_run()` produces:

- `research_run.json`
- `research_ledger.jsonl`
- `comparison.csv`
- `phase07_candidates.json`
- `rejected_configurations.json`

The human-readable report is `research_report.md`.

## Data Limitations

The report explicitly records missing volume, zero/default transaction costs and short samples. The current project's historical datasets are OHLC-focused; volume-derived features are disabled when volume is missing or incomplete.

## Phase Boundary

No optimizer is implemented here. Phase 06 discovers and documents research candidates. Parameter optimization, walk-forward selection, sensitivity and robustness studies belong to Phase 07.

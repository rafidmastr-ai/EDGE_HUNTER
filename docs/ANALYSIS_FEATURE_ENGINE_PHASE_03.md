# EDGE HUNTER — Phase 03 Analysis / Feature Engine

## Scope

Phase 03 provides a reusable OHLC-derived analysis layer. It does not contain Classic, SMC, ICT, entry, stop-loss, target, or trading-decision logic.

## Canonical outputs

- `FeatureDefinition` describes a feature, its group, source, lookback, and volume requirement.
- `MarketAnalysisSnapshot` stores feature values and diagnostics at one timestamp.
- `MarketAnalysisSeries` stores ordered snapshots, definitions, warm-up requirements, and engine metadata.

## Selected feature families

The initial feature set is deliberately broad but strategy-agnostic:

- Price/candle structure: range, body, body ratio, wicks, close location, candle direction.
- Trend: parameterized EMA values and an EMA alignment state.
- Momentum: Wilder RSI and MACD (line, signal, histogram).
- Volatility: Wilder ATR, ATR percent, and trailing standard deviation of one-bar log returns.
- Returns: one-bar log return plus parameterized simple returns.
- Volume: volume value/SMA/ratio only when every source bar has valid non-negative volume.

This is not a claim that this feature list is optimal for trading. Phase 06/07 research can evaluate or replace features using evidence.

## Warm-up behavior

Feature values are `None` until their own lookback is satisfied. The series also exposes `warmup_bars_required` and each snapshot exposes `warmup_complete` and available-history metadata.

## Look-ahead policy

The engine processes records strictly in timestamp order and never sorts or repairs the input. A duplicate or non-monotonic timestamp is rejected rather than silently changed.

The multi-timeframe aligner uses backward as-of matching: for a base timestamp `T`, it selects the latest source snapshot with timestamp `<= T`. `strict_before=True` changes this to `< T` when a caller requires an earlier source observation.

The tests deliberately append future bars and verify that earlier feature values remain unchanged.

## Volume policy

Volume-derived features are disabled when the dataset does not provide valid volume for every bar. No synthetic volume is created.

## Determinism and metadata

The engine is deterministic for identical canonical OHLC input and configuration. Each series contains an engine version, feature definitions, and per-snapshot metadata useful for later debugging and research.

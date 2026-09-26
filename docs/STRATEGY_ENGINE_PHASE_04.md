# EDGE HUNTER — Phase 04 Strategy Engine

## Scope

Phase 04 introduces a common, deterministic strategy framework while keeping strategy logic outside the generic feature layer.

Implemented families:

- Classic — `Classic_V1`
- SMC — `SMC_V1`
- ICT — `ICT_V1`

These are baseline, programmable variants for later research. Phase 04 does **not** claim profitability, superiority, or an optimal parameter set.

## Common contract

Every strategy receives `StrategyContext` and returns `StrategySignal`.

`StrategyContext.from_series(...)` slices both OHLC bars and feature snapshots through the decision index. This is the primary anti-look-ahead boundary for strategy evaluation.

A valid signal contains:

- decision timestamp
- symbol/timeframe
- direction
- entry
- stop loss
- exactly one target
- dynamic R:R
- entry/invalidation/SL/target logic
- evidence
- score inputs
- strategy name and variant
- inspectable metadata

Non-signal states are explicit: `NO_SIGNAL`, `CONFLICT`, and `INSUFFICIENT_DATA`.

## R:R policy

The shared `StrategyConfig` exposes a configurable `min_rr`/`max_rr` range. The Phase 04 default range is 1.5–2.0 as a transparent research parameter, not as a demonstrated optimum.

Strategies use available structural room to select an R:R inside that configured range. If sufficient room is unavailable, no trade signal is emitted.

## Classic_V1 exact rules

A BUY requires all of:

1. EMA20 > EMA50 > EMA200.
2. RSI >= 55.
3. MACD histogram > 0.
4. Decision close breaks above the preceding `swing_lookback` high.
5. Candle body ratio >= configured threshold.

SELL mirrors the conditions below the market.

Entry: decision-bar close.

Stop: opposite side of the pre-break recent range.

Target: one target using the dynamic structural-room R:R policy.

## SMC_V1 exact rules

A BUY requires:

1. Current low sweeps below the preceding local low and the close recovers above it.
2. Current close is above the previous close.
3. Candle body ratio meets the configured displacement threshold.
4. Current close breaks the previous candle high.

SELL mirrors the conditions using a high sweep, bearish displacement, and break below the previous candle low.

Entry: decision-bar close.

Stop: swept liquidity extreme.

Target: one target using the dynamic structural-room R:R policy.

This variant is an explicit testable SMC-inspired definition; it is not presented as the only definition of SMC.

## ICT_V1 exact rules

A BUY requires:

1. Three-candle bullish FVG: current low > high of the first candle in the three-candle sequence.
2. Current close > middle-candle close.
3. Current close > EMA20.
4. Candle body ratio meets the configured threshold.

SELL mirrors the FVG and displacement conditions.

Entry: decision-bar close.

Stop: local three-candle invalidation extreme.

Target: one target using the dynamic structural-room R:R policy.

Session-based killzones are intentionally not assumed in this baseline because session/timezone configuration belongs to later research/data policy.

## Extensibility

Future Hybrid/MTF/Experimental strategies implement the same `Strategy` protocol. No changes to the common signal contract are required.

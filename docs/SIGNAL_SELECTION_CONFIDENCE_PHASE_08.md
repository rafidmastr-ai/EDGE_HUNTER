# EDGE HUNTER — Phase 08: Signal Selection & Confidence Engine

## Scope

Phase 08 converts outputs from the common Strategy Engine contract into one
transparent, API-ready decision. All strategy outputs remain visible,
including weak and non-actionable outputs.

Implemented families:
- Classic
- SMC
- ICT

## Selection rules

1. Run every registered strategy using the supplied decision-time context.
2. Keep only actionable `SIGNAL` states for directional voting.
3. Count BUY and SELL votes equally by strategy.
4. Select a direction only if the leader reaches the configured
   `min_direction_share` and `min_direction_margin`.
5. A single strategy may be selected only when its calculated confidence
   reaches `min_single_signal_confidence`.
6. Otherwise return `NO_CLEAR_SIGNAL`.
7. Conflicting or insufficient strategies remain visible in the result.

These gates are engineering controls, not performance claims.

## Confidence

Confidence is a deterministic 0–100 evidence score. It is **not**:
- a guaranteed probability;
- a historical win rate;
- a prediction guarantee.

Default weights:
- strategy agreement: 20%
- signal quality: 15%
- structure/feature agreement: 10%
- entry quality: 10%
- SL/target quality: 10%
- R:R quality: 10%
- validated OOS performance: 10%
- sample size: 5%
- robustness: 5%
- market regime: 5%

Weights are configurable and must sum to 100%.

When validated OOS/sample/robustness/regime evidence is not available,
the corresponding component uses an explicit neutral 0.5 default and the
missing component is recorded in the result. No historical result is invented.

## One final target

The final target is copied from one representative selected strategy output.
The representative is deterministic: highest local evidence confidence,
then agreement, then strategy/variant name as tie-breakers.

No target averaging or multiple final targets are introduced.

## Arabic labels

Thresholds are configurable:
- ضعيف
- متوسط
- قوي
- قوي جدًا

The defaults are examples for implementation and are not universal optimal
thresholds.

## Calibration / monotonicity

`app/signals/calibration.py` provides optional diagnostics when actual outcome
labels are available:
- Brier score;
- confidence bins;
- observed win-rate gap;
- controlled monotonicity-violation count.

These diagnostics are evaluators, not hidden optimization.

## Look-ahead policy

The confidence layer does not access future OHLC history. It consumes only
strategy outputs generated from the already sliced decision-time context.
Identity validation also requires all strategy outputs to share one decision
timestamp, symbol and timeframe.

## API-ready output

`FinalSignalDecision.to_dict()` returns:
- timestamp/symbol/timeframe;
- final direction;
- confidence and Arabic label;
- Entry / SL / one Target / R:R;
- selected strategy + variant;
- traceable reasons;
- all strategy evaluations;
- directional support;
- scoring components;
- machine-readable metadata.

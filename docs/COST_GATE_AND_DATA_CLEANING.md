# Cost-aware gate and synthetic-bar cleaning

Two changes that came out of the EDGE HUNTER research phases (meta-labeling
research and the Market Edge Discovery study, EXP-001). Neither claims a new
trading edge; both remove systematic sources of loss/distortion.

## 1. Synthetic filler bars in historical CSV files

The historical files in `data/raw/*.csv` contain flat candles
(open = high = low = close = previous close) for periods when the market is
closed:

- every Sunday from 00:00 UTC until the real weekly open (~21:00/22:00 UTC),
- holidays (e.g. 24-25 Dec, 1 Jan),
- XAUUSD's daily maintenance break.

That is ~66,000-85,000 minutes per symbol per year. Indicators computed across
them (ATR, EMA, RSI, ...) are distorted after every re-open, and backtests /
learning treat them as tradable market data.

`app/data/cleaning.py::drop_synthetic_flat_runs` removes runs of flat bars that
repeat the last close (or follow a gap) and last at least 30 minutes and 3 bars.
Short quiet periods of real trading are kept. It is applied wherever historical
CSVs are read:

| Path | Used by |
|---|---|
| `app/data/pipeline.py` (`HistoricalDataPipeline.load`) | data pipeline; the count is reported as `synthetic_flat_bars_removed` |
| `app/learning/strategy_learning.py` (`read_ohlc_csv`) | `scripts/train_strategies.py` |
| `app/web/analysis_service.py` (`_read_csv`) | analysis in `local` mode |

Live provider data (Twelve Data) is not affected.

## 2. Cost-aware gate

On intraday timeframes the strategies' structure-based stops are often very
tight. The research measured the median round-trip trading cost at about
0.55-1.2 R per trade on M5 Forex and 0.30-0.65 R on M15. No setup quality can
recover that.

`app/signals/costs.py::CostGate` runs after the strategies (and after the
optional learned filter):

```
cost_r = estimated round-trip cost / |entry - stop_loss|
withhold the setup (NO_SIGNAL) if cost_r > max_cost_r      (default 0.10)
```

This is equivalent to requiring a stop at least `cost / max_cost_r` away
(10x the cost by default). The gate never changes prices. Accepted setups are
annotated with `cost_estimate`, `cost_r`, `min_stop_distance` and
`net_risk_reward_after_cost`. The API response metadata shows:

- `cost_gate`: the setting, plus the setups it withheld
- `trade_cost`: the cost annotation of the selected setup

Cost estimates (`CostModel`) are conservative assumptions, not broker quotes:

- **Forex:** spread by pair + 0.3 pip slippage + 0.7 pip commission (EURUSD
  1.8 pips, NZDUSD 2.8 pips, other pairs 3.5 pips).
- **Metals:** fixed estimate (XAU 0.42 USD).
- **Crypto:** 0.10% of price.
- **Other instruments:** 0.05% of price.

Configuration:

```
EDGE_HUNTER_COST_GATE_ENABLED=true
EDGE_HUNTER_COST_GATE_MAX_COST_R=0.10
```

The gate applies in both `live` and `local` modes. It is a deterministic rule,
not a learned model. Raw strategy setups are still recorded for learning.

Research note: in the meta-labeling study a cost-only filter chosen on
development data matched or beat the ML filter on most strategy x timeframe
combinations. Its confidence intervals still included zero, so treat it as risk
hygiene, not as proven profitability.

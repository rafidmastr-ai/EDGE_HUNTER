# Strategy Machine Learning (Classic / SMC / ICT)

## What it does

Each rule-based strategy keeps generating its own setups. A per-strategy model
(logistic regression, meta-labeling) learns the probability that a setup reaches
its target before its stop-loss. During analysis, a setup whose probability is
below the strategy's learned threshold is withheld (`ml_filter_rejected`); the
others pass through unchanged, annotated with `ml_probability`. Trade prices,
risk and lot size are never changed by the model.

## Learning sources

1. **Historical CSV files** in `data/raw` (file name = symbol, e.g. `BTCUSD.csv`;
   ISO timestamps or Unix epoch seconds/milliseconds). Every strategy is replayed
   on M5 / M15 / H1 and every setup is labelled with the Phase 05 TP/SL semantics
   (`OutcomeLabelEngine`, stop-first on ambiguous candles, horizon
   `EDGE_HUNTER_STRATEGY_ML_HORIZON_BARS`).
2. **Live setups**: after `/api/analyze` has sent its response (FastAPI background
   task), each actionable strategy setup is stored as a `LIVE_TRADE`
   `LearningRecord` (`PENDING_OUTCOME`). Recording never delays the response.
   The training script later labels them with the candles that followed
   (fetched from Twelve Data) and includes them in training.

## Run

```text
python scripts/train_strategies.py
```

Options: `--data-dir PATH`, `--max-bars-per-timeframe N`, `--horizon-bars N`,
`--skip-live`. The models are written to `data/models/strategy_learning.json`
(with a SHA-256 checksum; no pickle) and a report to
`data/models/strategy_learning_report.txt`. The running web app reloads the
model file automatically within about 30 seconds; no restart is needed.

### MT5 history folders and a fixed test period

A sub-folder of `data/raw` with a `SOURCE.json` (e.g. `data/raw/XAUUSD/`) is read
as an MT5 M1 export: all `<SYM>_M1_<YEAR>.csv` files are merged, and the
timestamps (server time written with a misleading `+00:00`) are converted to
UTC with the `server_tz` named in `SOURCE.json` (`app/data/mt5_history.py`).

```text
python scripts/train_strategies.py --skip-live --symbols XAUUSD --timeframes H1,M15,M5 \
    --test-fraction 0.2 --test-end 2025-01-01 --cost XAUUSD=0.20 --out data/models/xauusd_split
```

- TEST = the last `--test-fraction` of the symbol's bars before `--test-end`
  (for XAUUSD: 2023-11-14 → 2024-12-31). Everything else is used for learning:
  in each calendar year the last 20% of setups is VALIDATION (threshold choice),
  the rest TRAIN.
- The bar series is cut at the test boundaries and each piece is replayed on
  its own, so no label horizon or indicator warm-up crosses between test and
  training data.
- `--cost SYMBOL=PRICE` charges a round-trip cost of `PRICE / |entry - stop|` R
  on every labelled setup.
- `--out DIR` writes the model and report to `DIR`; the live model file is not
  touched. The report adds a table per timeframe for TRAIN / VALIDATION / TEST,
  all setups vs model-filtered: setups, win rate, average and total R, profit
  factor, max drawdown (R, per setup) and a 95% block-bootstrap interval of the
  average R.

## Promotion gate

Samples are split chronologically: TRAIN 60% / VALIDATION 20% / OOS 20%. The
scaler and model are fitted on TRAIN only and the threshold is chosen on
VALIDATION among quantiles of the predicted probabilities (keep the top 100%,
95%, ... 20% of setups), so strategies with a low base win rate are handled.
The report also lists each strategy's unfiltered results per symbol. A model becomes `ACTIVE` only if it raises the average R per setup by
at least 0.02R on both VALIDATION and the untouched OOS segment, while keeping
enough setups. Otherwise the model is `REJECTED` (or `INSUFFICIENT_DATA`) and the
strategy runs exactly as before. Status is visible in `/api/meta` →
`strategy_learning` and in each analysis response metadata.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `EDGE_HUNTER_STRATEGY_ML_ENABLED` | `true` | apply ACTIVE models in the web analysis |
| `EDGE_HUNTER_STRATEGY_ML_RECORD_LIVE` | `true` | record live setups (live mode only) after each response |
| `EDGE_HUNTER_STRATEGY_ML_HORIZON_BARS` | `48` | candles a setup may take to hit TP/SL before it counts as expired |
| `EDGE_HUNTER_STRATEGY_ML_MODEL_PATH` | `data/models/strategy_learning.json` | model file |

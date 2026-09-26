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

## Promotion gate

Samples are split chronologically: TRAIN 60% / VALIDATION 20% / OOS 20%. The
scaler and model are fitted on TRAIN only and the threshold is chosen on
VALIDATION. A model becomes `ACTIVE` only if it raises the average R per setup by
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

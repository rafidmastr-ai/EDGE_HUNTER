# EDGE HUNTER — Phase 07
## Optimization, Validation & Robustness

Phase 07 improves strategy configurations without treating one backtest result as proof of quality.

### Search
- Grid and deterministic Random search are implemented.
- The default search space uses only dimensions explicitly explored in Phase 06: R:R bounds, candle-body confirmation, and swing lookback.
- The search stage scores TRAIN data only.

### Validation
- Chronological Train / Validation / OOS split.
- Validation gates include sample size, expectancy, profit factor, stability and optional drawdown.
- OOS remains held out evidence and is not used to generate the search space.

### Robustness
- Local parameter sensitivity.
- Small-parameter-change deterioration detection.
- Walk-forward fold evaluation.
- Optional cross-case/cross-symbol checks.
- Parameter instability flags are explicit.

### Reproducibility
- Search method and seed are stored.
- Dataset hash, split plan, strategy configuration and evaluation IDs are persisted.
- Outputs: JSON, JSONL, CSV and Markdown.

### Policy
A higher net return alone does not determine a final candidate. Validated candidates must remain coherent across validation, OOS and robustness checks.

Run smoke optimization:

```powershell
python scripts/run_phase07_optimization.py
```

Run the complete development checks through:

```powershell
python main.py
```

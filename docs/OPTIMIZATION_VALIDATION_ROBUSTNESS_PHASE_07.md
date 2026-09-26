# Phase 07 — Optimization, Validation & Robustness

## Objective
Improve documented strategy configurations while reducing overfit risk.

## Search policy
Grid and deterministic Random search are implemented first. Bayesian/genetic methods are deliberately not included because the current project does not yet require an additional dependency or larger search budget. The architecture leaves room for future search adapters.

Only parameters with a Phase 06 research reason enter the default search space:
- `min_rr` / `max_rr`: Phase 06 tested fixed 1.50, 1.75, 2.00 and dynamic 1.50–2.00 behavior.
- `minimum_body_ratio`: Phase 06 tested 0.55 and 0.65.
- `swing_lookback`: Phase 06 tested 5 and 8.

## Chronological data separation
Feature calculations are built on the full chronological dataset, while each backtest period is window-gated. This preserves valid historical warm-up and prevents future-period trades from contaminating training.

- TRAIN: search/parameter scoring only.
- VALIDATION: candidate admission.
- OOS: held-out evidence only.
- Walk-forward: repeated forward test windows.

## Objective policy
The optimization search score combines expectancy, profit factor, drawdown behavior, stability, sample sufficiency, average R:R and win-rate balance. Net return is not the sole objective.

## Sensitivity
For every retained search configuration the framework compares nearby configurations within Hamming distance 1 across the evaluated parameter space. Large local deterioration yields `PARAMETER_SENSITIVE`; low sensitivity score can yield `PARAMETER_UNSTABLE`.

## Robustness
Optional cross-case checks can evaluate a configuration on additional symbols/datasets. Walk-forward tests are run when enough bars exist. The framework records positive-case and positive-fold ratios.

## Reproducibility
Every run records:
- engine/version
- strategy target
- search space
- seed
- chronological split
- dataset SHA-256
- configuration IDs
- evaluation outputs

## Outputs
`optimization_run.json`, `optimization_evaluations.jsonl`, `validated_candidates.json`, `sensitivity_report.csv`, and a Markdown report.

## Acceptance mapping
The Phase 07 tests cover deterministic search, chronological splits, multi-objective scoring, OOS separation, persistence, sensitivity, parameter instability, walk-forward structure and reproducibility.

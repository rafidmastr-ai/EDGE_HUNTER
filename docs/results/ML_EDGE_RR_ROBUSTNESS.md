# Clean R:R robustness test — FAR_TARGET only

**Files**
- Full tables: `ML_EDGE_RR_ROBUSTNESS.txt`, `ML_EDGE_RR_ROBUSTNESS.json`
- Run: `python scripts/ml_edge_rr_robustness.py`

## Setup
**Fixed:**
- the saved B1 EURJPY model and its signals;
- entry at the next minute's open after the 15-minute decision bar;
- **SL = 2 × ATR14(M15)**;
- the direction/filter rule;
- spread 0.1 pip, swap 0.5 pip/night (Wednesday ×3), 12 h cap, Friday 20:45 UTC exit.

**Changed:** only the target, TP = R:R × SL, for 1:1, 1:1.5, … 1:6.

**Nothing was retrained, tuned or selected.**

**Forward sets (evaluation only):**
- EURJPY 2025-09-01 → 2026-09-28
- AUDJPY 2026-04 → 2026-09-28 (unseen pair)
- CADJPY 2026-04 → 2026-09-28 (unseen pair)

**Rules:**
- **Primary:** the current rule (keep 5%, both directions).
- **Secondary:** strict sell (keep 2%, sell only). This set is small, 23–50 trades.

**95% CI:** percentile bootstrap of the mean R (5,000 resamples), using the same method for every row.

**Stability criterion (declared before running):** an R:R counts as generalizable only if avg R > 0 **and** the 95% CI lower bound > 0 on **all three** pairs.

## Primary rule — avg R per trade (net R) and 95% CI lower bound

| R:R | EURJPY | AUDJPY | CADJPY | Passes on all 3? |
|---|---|---|---|---|
| 1:1 | +0.193 (+44.7), CI low +0.07 | +0.450 (+42.3), +0.28 | +0.248 (+28.8), +0.07 | **yes** |
| 1:1.5 | +0.121 (+25.2), −0.04 | +0.609 (+51.8), +0.37 | +0.311 (+31.8), +0.08 | no (EURJPY CI) |
| 1:2 | +0.026 (+5.0), −0.15 | +0.660 (+54.8), +0.37 | +0.321 (+30.8), +0.05 | no |
| 1:2.5 | −0.015 (−3.0), −0.21 | +0.823 (+61.8), +0.48 | +0.207 (+18.2), −0.09 | no |
| 1:3 | +0.009 (+1.7), −0.20 | +0.931 (+68.9), +0.53 | +0.187 (+16.3), −0.13 | no |
| 1:4 | +0.026 (+4.7), −0.19 | +0.937 (+68.4), +0.49 | +0.149 (+12.7), −0.16 | no |
| 1:5 | +0.033 (+6.0), −0.20 | +0.994 (+70.6), +0.50 | +0.116 (+9.8), −0.20 | no |
| 1:6 | +0.031 (+5.6), −0.20 | +0.914 (+64.9), +0.44 | +0.078 (+6.6), −0.22 | no |

**Exits (TP / SL / time)** on EURJPY:

| R:R | TP | SL | Time |
|---|---|---|---|
| 1:1 | 132 | 83 | 17 |
| 1:2 | 49 | 102 | 41 |
| 1:3 | 28 | 109 | 52 |
| 1:4 | 16 | 106 | 62 |
| 1:6 | 5 | 106 | 73 |

**Secondary rule (strict sell):** the same picture.
- Only 1:1 passes on all three pairs.
- EURJPY is negative from 1:2.5 up.
- AUDJPY peaks around 1:2–1:2.5, and CADJPY peaks around 1:1.5–1:2.

## Cross-pair comparison
1. **The pairs disagree on the shape of the curve.**
   - **EURJPY** falls steadily from 1:1. Every higher R:R loses 20–48 R versus 1:1.
   - **AUDJPY** rises up to about 1:3 and then stays flat around +0.9 to +1.0 R.
   - **CADJPY** peaks at 1:1.5–1:2 and then declines. From 1:2.5 up, its CI includes zero.
   - Rank correlation of the avg-R curves between pairs:
     - EURJPY ~ AUDJPY: −0.21
     - EURJPY ~ CADJPY: +0.11
     - AUDJPY ~ CADJPY: −0.80
   - There is no shared optimum. Each pair's "best" R:R (1:1, 1:5, 1:2) is an artefact of that pair and period.
2. **Why the curves go flat above about 1:3:** the target is almost never reached within the 12 h cap.
   - On EURJPY only 5 of 184 trades hit TP at 1:6.
   - Win rate stops changing (35.9% from 1:3.5 to 1:6) because trades end by the stop or by time.
   - 1:3 … 1:6 are therefore effectively **one strategy**: "hold up to 12 h unless stopped".
   - Their results depend on whether the pair trended during the window, as AUDJPY did in 2026, not on the R:R.
3. **Minimum across pairs** (worst-case avg R): +0.19 at 1:1, +0.12 at 1:1.5, then about 0 (−0.02 to +0.04) for every R:R from 1:2 to 1:6.

## Conclusion
- **The only R:R that generalizes under the pre-declared test is 1:1**, under both rules.
- **1:1.5** is positive on all three pairs, but its EURJPY CI crosses zero (−0.04), so it is at most a weak "maybe". The evidence for it is weaker, not stronger.
- **There is no stable range above 1:1.5.** Higher-R:R gains appear only on AUDJPY, whose 2026 move was unusually strong. On EURJPY (the longest set) and on CADJPY they turn into losses relative to 1:1.
- This matches how the model was built: it was trained to predict a ±2 ATR move with a 1:1 barrier. Far targets ask it for something it never learned.

**Caveats:**
- AUDJPY and CADJPY cover about 6 months in the same yen regime and are correlated with EURJPY, so the three pairs are not fully independent evidence.
- Costs are demo-level (0.1 pip).

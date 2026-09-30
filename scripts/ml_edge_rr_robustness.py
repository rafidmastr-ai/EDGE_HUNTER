"""Clean R:R robustness test (FAR_TARGET only) for the B1 EURJPY model on yen pairs.

Everything fixed: saved model and signals, entry (next minute open), SL = 2 x ATR14(M15),
direction/filter rule, costs (0.1 pip), swap (0.5 pip/night), 12 h cap, Friday 20:45 UTC exit.
Only the target changes: TP = RR x SL for RR in 1, 1.5, ..., 6. Nothing is chosen or tuned.

Forward sets (evaluation only): EURJPY 2025-09-01 -> 2026-09-28; AUDJPY and CADJPY
2026-04 -> 2026-09-28 (pairs never seen by the model).

Stability criteria, declared before running:
  - an R:R is "generalizable" only if, on ALL three pairs, avg R > 0 AND the 95 % CI lower bound > 0;
  - the shape of avg R vs R:R is compared across pairs (rank correlation), no single best value is picked.

    python scripts/ml_edge_rr_robustness.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rrmod = _load("ml_edge_rr_jpy")
imp = rrmod.imp

from app.ml_edge.data import TRAINING_SYMBOLS, load_symbol  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.walkforward import DEV_END  # noqa: E402

RESULTS = PROJECT_ROOT / "docs" / "results"
RRS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
RULES = {
    "primary: current rule (keep 5%, both directions)": {"keep": 0.05, "side": "both", "breakeven": None},
    "secondary: strict sell (keep 2%, sell only)": {"keep": 0.02, "side": "sell", "breakeven": None},
}
PAIRS = ("EURJPY", "AUDJPY", "CADJPY")


def bootstrap_ci(r: np.ndarray, runs: int = 5000, seed: int = 0) -> list[float] | None:
    """Percentile bootstrap 95 % CI of the mean R (same method for every row, any sample size)."""
    if len(r) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, len(r), size=(runs, len(r)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    symbols = {s: load_symbol(imp.RAW / s, imp.CACHE) for s in TRAINING_SYMBOLS}
    for p in imp.NEW_PAIRS:
        symbols[p] = load_symbol(imp.RAW / p)
    frames = build_all(symbols, session_start_hour=1)
    forward = {"EURJPY": (imp.Series("EURJPY", symbols, frames, DEV_END), (imp.SELECTION_END, 2**62), "2025-09-01 -> 2026-09-28")}
    for p in imp.NEW_PAIRS:
        start = int(frames[p].close_time[imp.FULL_WINDOW_BARS])
        forward[p] = (imp.Series(p, symbols, frames, start), (0, 2**62), "2026-04 -> 2026-09-28 (unseen pair)")

    report: dict = {"rr": RRS, "sl": "2 x ATR14(M15)", "periods": {p: v[2] for p, v in forward.items()}, "rules": {}}
    for rule_name, rule in RULES.items():
        table = {p: {} for p in PAIRS}
        for pair, (series, window, _) in forward.items():
            for rr in RRS:
                tr = series.run({**rule, "stop_atr": 2.0, "target_rr": rr}, window)
                s = rrmod.stats(tr, pair, rr)
                s["exit_time"] = s["exit_time_12h"] + s["exit_time_friday"]
                s["avg_r_ci95"] = bootstrap_ci(tr.r_net)
                table[pair][f"1:{rr:g}"] = s
        keys = [f"1:{rr:g}" for rr in RRS]
        avg = {p: np.array([table[p][k]["avg_r"] for k in keys]) for p in PAIRS}
        lo = {p: np.array([(table[p][k]["avg_r_ci95"] or [np.nan])[0] for k in keys]) for p in PAIRS}
        analysis = {
            "generalizable_rr (all pairs avg>0 and CI low>0)": [k for i, k in enumerate(keys) if all(avg[p][i] > 0 and lo[p][i] > 0 for p in PAIRS)],
            "positive_on_all_pairs": [k for i, k in enumerate(keys) if all(avg[p][i] > 0 for p in PAIRS)],
            "min_avg_r_across_pairs": {k: round(float(min(avg[p][i] for p in PAIRS)), 4) for i, k in enumerate(keys)},
            "best_rr_per_pair (information only, not a choice)": {p: keys[int(np.argmax(avg[p]))] for p in PAIRS},
            "rank_correlation_of_avg_r_curves": {f"{a}~{b}": round(spearman(avg[a], avg[b]), 3)
                                                 for i, a in enumerate(PAIRS) for b in PAIRS[i + 1:]},
            "change_vs_1to1_net_r": {p: {k: round(table[p][k]["net_r"] - table[p]["1:1"]["net_r"], 1) for k in keys[1:]} for p in PAIRS},
        }
        report["rules"][rule_name] = {"table": table, "analysis": analysis}
        print(rule_name, json.dumps(analysis, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ML_EDGE_RR_ROBUSTNESS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_txt(report, RESULTS / "ML_EDGE_RR_ROBUSTNESS.txt")
    return 0


def write_txt(report: dict, path: Path) -> None:
    L = ["EDGE HUNTER - clean R:R robustness test, FAR_TARGET only (B1 EURJPY model, saved, unchanged)", "=" * 132,
         "Fixed: signals, entry (next minute open after the 15-min decision bar), SL = 2 x ATR14(M15), direction/filter rule,",
         "       spread 0.1 pip, swap 0.5 pip/night (Wed x3), 12 h cap, Friday 20:45 UTC exit.  Changed: TP = RR x SL only.",
         "Forward sets (evaluation only, nothing tuned): " + "; ".join(f"{p} {v}" for p, v in report["periods"].items()),
         "Stability rule (declared before running): R:R generalizable only if avg R > 0 and 95% CI low > 0 on ALL three pairs.",
         "95% CI: percentile bootstrap of the mean R (5,000 resamples, fixed seed), same method for every row.", ""]
    for rule_name, block in report["rules"].items():
        L.append("#" * 132)
        L.append(rule_name)
        for pair, rows in block["table"].items():
            L.append(f"  {pair} {report['periods'][pair]}")
            L.append(f"    {'R:R':<6}{'trades':>7}{'win%':>7}{'need%':>7}{'avgR':>8}{'PF':>7}{'netR':>8}{'maxDD':>7}{'TP':>5}{'SL':>5}{'time':>6}   95% CI of avg R")
            for k, s in rows.items():
                ci = s["avg_r_ci95"]
                ci_text = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
                L.append(f"    {k:<6}{s['trades']:>7}{100 * s['win_rate']:>7.1f}{100 * s['breakeven_win_rate_needed']:>7.1f}{s['avg_r']:>+8.3f}"
                         f"{(s['profit_factor'] or 0):>7.2f}{s['net_r']:>+8.1f}{s['max_drawdown_r']:>7}{s['exit_tp']:>5}{s['exit_sl']:>5}{s['exit_time']:>6}"
                         f"   {ci_text}")
        a = block["analysis"]
        L.append("  CROSS-PAIR ANALYSIS")
        for k, v in a.items():
            L.append(f"    {k}: {v}")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

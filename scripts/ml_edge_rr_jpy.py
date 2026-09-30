"""R:R from 1:1.4 upward for the B1 EURJPY model on yen pairs (execution only; the model is unchanged).

Two ways to build 1:RR:
  far_target : stop stays 2 x ATR14(M15), target = RR x stop (2.8 ATR for 1:1.4 ... 6 ATR for 1:3)
  close_stop : target stays 2 x ATR (the distance the model was trained to predict), stop = 2 x ATR / RR
1:1 (the current rule) is the reference. 12 h cap, Friday 20:45 UTC exit, spread 0.1 pip,
swap 0.5 pip/night as before. Rows are compared on EURJPY 2023-11 -> 2025-08; the other sets check.

    python scripts/ml_edge_rr_jpy.py
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

_spec = importlib.util.spec_from_file_location("ml_edge_improve_jpy", Path(__file__).with_name("ml_edge_improve_jpy.py"))
imp = importlib.util.module_from_spec(_spec)
sys.modules["ml_edge_improve_jpy"] = imp
_spec.loader.exec_module(imp)

from app.ml_edge.data import TRAINING_SYMBOLS, cost_price, load_symbol  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.walkforward import DEV_END  # noqa: E402
from app.research.intraday import DAY, full_metrics  # noqa: E402

RESULTS = PROJECT_ROOT / "docs" / "results"
RRS = (1.0, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
TRADE_LIST_RR = 1.4
RULES = {
    "current (keep 5%, both directions)": {"keep": 0.05, "side": "both", "breakeven": None},
    "strict sell (keep 2%, sell only)": {"keep": 0.02, "side": "sell", "breakeven": None},
}
RANDOM_RUNS = 200


def longest_losing_streak(r: np.ndarray) -> int:
    best = run = 0
    for v in r:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


def case(family: str, rr: float) -> dict:
    if family == "far_target":
        return {"stop_atr": 2.0, "target_rr": float(rr)}
    return {"stop_atr": 2.0 / rr, "target_rr": float(rr)}


def stats(tr, pair: str, rr: float) -> dict:
    if not len(tr):
        return {"trades": 0}
    order = np.argsort(tr.entry_time, kind="stable")
    r = tr.r_net[order]
    m = full_metrics(tr)
    sod = tr.exit_time % DAY
    friday = ((tr.exit_time // DAY + 3) % 7 == 4) & (sod >= 20 * 3600 + 44 * 60)
    risk = np.abs(tr.entry - tr.stop)
    return {
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean()), 4),
        "breakeven_win_rate_needed": round(1 / (1 + rr), 4),
        "losing_trades": int((r < 0).sum()),
        "avg_r": round(float(r.mean()), 4),
        "median_r": round(float(np.median(r)), 4),
        "profit_factor": m.get("profit_factor"),
        "net_r": round(float(r.sum()), 2),
        "avg_r_ci95": m.get("avg_r_ci95"),
        "max_drawdown_r": m.get("max_drawdown_r"),
        "longest_losing_streak": longest_losing_streak(r),
        "exit_tp": int((tr.outcome == 1).sum()),
        "exit_sl": int((tr.outcome == -1).sum()),
        "exit_time_12h": int(((tr.outcome == 0) & ~friday).sum()),
        "exit_time_friday": int(((tr.outcome == 0) & friday).sum()),
        "avg_spread_cost_r": round(float(np.mean(cost_price(pair) / risk)), 4),
        "avg_hold_minutes": round(float(np.mean(tr.exit_time - tr.entry_time) / 60), 1),
        # fills do not depend on the cost, so a wider spread is an exact re-costing of the same trades
        **{f"avg_r_spread_{level}": round(float(np.mean(tr.r_net - (cost_price(pair, level) - cost_price(pair)) / risk)), 4)
           for level in ("sens_mid", "sens_high")},
    }


def main() -> int:
    symbols = {s: load_symbol(imp.RAW / s, imp.CACHE) for s in TRAINING_SYMBOLS}
    for p in imp.NEW_PAIRS:
        symbols[p] = load_symbol(imp.RAW / p)
    frames = build_all(symbols, session_start_hour=1)
    ej = imp.Series("EURJPY", symbols, frames, DEV_END)
    sets = {
        "EURJPY 2023-11 -> 2025-08 (comparison set)": (ej, (DEV_END, imp.SELECTION_END)),
        "EURJPY 2025-09 -> 2026-09": (ej, (imp.SELECTION_END, 2**62)),
    }
    for p in imp.NEW_PAIRS:
        start = int(frames[p].close_time[imp.FULL_WINDOW_BARS])
        sets[f"{p} 2026-04 -> 2026-09 (unseen pair)"] = (imp.Series(p, symbols, frames, start), (0, 2**62))

    rng = np.random.default_rng(9)
    report: dict = {"rr": RRS, "rules": RULES, "families": {
        "far_target": "stop 2 x ATR, target RR x stop", "close_stop": "target 2 x ATR, stop 2 x ATR / RR"}, "results": {}}
    trade_rows = []
    for set_name, (series, window) in sets.items():
        report["results"][set_name] = {}
        for rule_name, rule in RULES.items():
            rows = {}
            for family in ("far_target", "close_stop"):
                for rr in RRS:
                    if rr == 1.0 and family == "close_stop":
                        continue  # identical to far_target 1:1
                    cfg = {**rule, **case(family, rr)}
                    tr = series.run(cfg, window)
                    s = stats(tr, series.pair, rr)
                    d = series.direction(rule["keep"])
                    rand = []
                    for _ in range(RANDOM_RUNS // 4):
                        signs = np.where(rng.random(len(d)) < 0.5, 1, -1) * (d != 0)
                        rt = series.run({**cfg, "side": "both"}, window, direction=signs)
                        rand.append(float(rt.r_net.mean()) if len(rt) else np.nan)
                    s["random_direction_mean_avg_r"] = round(float(np.nanmean(rand)), 4)
                    s["random_runs_beating"] = round(float(np.mean(np.array(rand) >= s.get("avg_r", 0))), 3)
                    key = "1:1 (current)" if rr == 1 else f"{family} 1:{rr:g}"
                    rows[key] = s
                    if rr == TRADE_LIST_RR:
                        for i in range(len(tr)):
                            trade_rows.append((set_name, rule_name, key, tr, i, cfg))
            report["results"][set_name][rule_name] = rows
            print(set_name, "|", rule_name)
            for k, s in rows.items():
                print(f"   {k:<20} n {s['trades']:>4} win {s['win_rate']:.2f} (need {s['breakeven_win_rate_needed']:.2f}) "
                      f"lose {s['losing_trades']:>4} avgR {s['avg_r']:+.3f} net {s['net_r']:+7.1f} DD {s['max_drawdown_r']} "
                      f"streak {s['longest_losing_streak']} TP/SL/12h/Fri {s['exit_tp']}/{s['exit_sl']}/{s['exit_time_12h']}/{s['exit_time_friday']} "
                      f"cost {s['avg_spread_cost_r']} rand {s['random_runs_beating']} R@0.5pip {s['avg_r_spread_sens_mid']:+.3f} R@1.5pip {s['avg_r_spread_sens_high']:+.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ML_EDGE_JPY_RR.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_txt(report, trade_rows, RESULTS / "ML_EDGE_JPY_RR_TRADES.txt")
    return 0


def write_txt(report: dict, rows: list, path: Path) -> None:
    L = ["EDGE HUNTER - B1 EURJPY model on yen pairs: R:R 1:1 vs 1:1.4 ... 1:3 (research only)", "=" * 130,
         "far_target : stop 2 x ATR14(M15), target RR x stop.   close_stop : target 2 x ATR, stop 2 x ATR / RR.",
         "Entry next minute open after the 15-minute decision bar; 12 h cap; Friday 20:45 UTC exit; spread 0.1 pip; swap 0.5 pip/night (Wed x3).",
         "win% need = break-even win rate 1/(1+RR) before costs.  exits TP/SL/12h/Fri.  streak = longest run of losing trades.",
         "R@0.5p / R@1.5p = average R of the same trades with a 0.5 / 1.5 pip spread (demo spread is 0.1 pip).", ""]
    for set_name, rules in report["results"].items():
        L.append(set_name)
        for rule_name, rows_ in rules.items():
            L.append(f"  {rule_name}")
            L.append(f"    {'case':<20}{'trades':>7}{'win%':>7}{'need%':>7}{'losers':>8}{'avgR':>8}{'medR':>8}{'PF':>7}{'netR':>8}{'maxDD':>7}{'streak':>7}{'TP':>5}{'SL':>5}{'12h':>5}{'Fri':>5}{'costR':>7}{'rand%':>7}{'R@0.5p':>8}{'R@1.5p':>8}")
            for k, s in rows_.items():
                L.append(f"    {k:<20}{s['trades']:>7}{100 * s['win_rate']:>7.1f}{100 * s['breakeven_win_rate_needed']:>7.1f}{s['losing_trades']:>8}"
                         f"{s['avg_r']:>+8.3f}{s['median_r']:>+8.3f}{(s['profit_factor'] or 0):>7.2f}{s['net_r']:>+8.1f}{s['max_drawdown_r']:>7}"
                         f"{s['longest_losing_streak']:>7}{s['exit_tp']:>5}{s['exit_sl']:>5}{s['exit_time_12h']:>5}{s['exit_time_friday']:>5}"
                         f"{s['avg_spread_cost_r']:>7.3f}{100 * s['random_runs_beating']:>7.0f}{s['avg_r_spread_sens_mid']:>+8.3f}{s['avg_r_spread_sens_high']:>+8.3f}")
        L.append("")
    L.append(f"TRADES at 1:{TRADE_LIST_RR:g} (both families, both rules)")
    L.append(f"{'#':>4} {'set':<44} {'rule':<34} {'case':<18} {'entry (UTC)':<16} {'dir':<4} {'entry':>9} {'SL':>9} {'TP':>9} {'exit (UTC)':<16} {'exit':>9} {'why':<5} {'R_net':>7}")
    for k, (set_name, rule_name, key, tr, i, cfg) in enumerate(rows, 1):
        d = int(tr.direction[i])
        risk = abs(tr.entry[i] - tr.stop[i])
        tp = tr.entry[i] + d * cfg["target_rr"] * risk
        why = {1: "TP", -1: "SL"}.get(int(tr.outcome[i]), "TIME")
        L.append(f"{k:>4} {set_name[:44]:<44} {rule_name[:34]:<34} {key:<18} {imp.iso(tr.entry_time[i]):<16} {'BUY' if d > 0 else 'SELL':<4} "
                 f"{tr.entry[i]:>9.3f} {tr.stop[i]:>9.3f} {tp:>9.3f} {imp.iso(tr.exit_time[i]):<16} {tr.exit[i]:>9.3f} {why:<5} {tr.r_net[i]:>+7.3f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

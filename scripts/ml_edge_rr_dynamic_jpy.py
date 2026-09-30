"""R:R 1:4.1 -> 1:6 for the B1 EURJPY model on yen pairs: fixed-ratio scan (step 0.1) and dynamic per-trade ratios.

Constructions (as in ml_edge_rr_jpy.py):
  far_target : stop 2 x ATR14(M15), target RR x stop
  close_stop : target 2 x ATR, stop 2 x ATR / RR
Dynamic rules: each trade gets RR = 4.1 + 1.9 x p, where p is a causal percentile (0..1) of a
driver among the previous 60 days of decision bars (no future data):
  strength      : |model forecast|   (stronger signal -> larger RR)
  strength_inv  : 1 - p of |forecast|
  volatility    : vol_level_z feature (higher volatility -> larger RR)
  volatility_inv: 1 - p of vol_level_z
The best rule (fixed ratio or dynamic) is chosen per construction and entry rule on
EURJPY 2023-11 -> 2025-08 only (net R / max drawdown, >= 50 trades), then checked on
EURJPY 2025-09 -> 2026-09 and the unseen AUDJPY / CADJPY (2026-04 -> 2026-09).

    python scripts/ml_edge_rr_dynamic_jpy.py
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
from app.ml_edge.walkforward import DEV_END, execution_config  # noqa: E402
from app.research.intraday import DAY, make_orders, simulate  # noqa: E402

RESULTS = PROJECT_ROOT / "docs" / "results"
FIXED = tuple(round(4.1 + 0.1 * k, 1) for k in range(20))  # 4.1 .. 6.0
DYNAMIC = ("strength", "strength_inv", "volatility", "volatility_inv")
RR_LOW, RR_HIGH = 4.1, 6.0
FAMILIES = ("far_target", "close_stop")
RULES = rrmod.RULES
COMPARISON = "EURJPY 2023-11 -> 2025-08 (comparison set)"
MIN_TRADES = 50
RANDOM_RUNS = 100


def causal_percentile(times: np.ndarray, values: np.ndarray, window_days: int = 60, min_days: int = 20) -> np.ndarray:
    """Percentile of each value among the previous ``window_days`` of values (strictly earlier days)."""
    days = times // DAY
    out = np.full(len(values), 0.5)
    for day in np.unique(days):
        past = values[(days < day) & (days >= day - window_days) & np.isfinite(values)]
        if len(np.unique(days[(days < day) & (days >= day - window_days)])) < min_days or not len(past):
            continue
        past = np.sort(past)
        today = days == day
        v = values[today]
        out[today] = np.where(np.isfinite(v), np.searchsorted(past, v, "right") / len(past), 0.5)
    return out


class DynamicSeries:
    def __init__(self, series, frame_names) -> None:
        self.s = series
        _, forecast = series.model.decide(series.times, series.x, series.atr)
        self.p = {
            "strength": causal_percentile(series.times, np.abs(forecast)),
            "volatility": causal_percentile(series.times, series.x[:, frame_names.index("vol_level_z")].astype(float)),
        }

    def rr_array(self, how) -> np.ndarray:
        n = len(self.s.times)
        if isinstance(how, float):
            return np.full(n, how)
        base, inverse = how.replace("_inv", ""), how.endswith("_inv")
        p = 1 - self.p[base] if inverse else self.p[base]
        return RR_LOW + (RR_HIGH - RR_LOW) * np.clip(p, 0, 1)

    def run(self, rule: dict, family: str, how, window, direction=None):
        s = self.s
        d = s.direction(rule["keep"]).copy() if direction is None else direction.copy()
        if rule["side"] == "sell":
            d[d > 0] = 0
        elif rule["side"] == "buy":
            d[d < 0] = 0
        rr = self.rr_array(how)
        sel = (d != 0) & (s.times >= window[0]) & (s.times < window[1])
        stop_atr = np.full(len(rr), 2.0) if family == "far_target" else 2.0 / rr
        n = int(sel.sum())
        orders = make_orders(s.times[sel], d[sel], np.full(n, np.nan), np.full(n, np.nan),
                             stop_distance=stop_atr[sel] * s.atr[sel], target_rr=rr[sel])
        tr = simulate(s.data.m1, orders, execution_config(s.pair, s.model.label))
        rr_used = dict(zip(s.times[sel].tolist(), rr[sel].tolist()))
        return tr, np.array([rr_used[int(t)] for t in tr.signal_time])


def summarize(tr, pair, rr_used) -> dict:
    if not len(tr):
        return {"trades": 0}
    out = rrmod.stats(tr, pair, float(np.mean(rr_used)))
    out["avg_rr"] = round(float(np.mean(rr_used)), 2)
    out["breakeven_win_rate_needed"] = round(float(np.mean(1 / (1 + rr_used))), 4)
    out["ret_over_dd"] = round(out["net_r"] / out["max_drawdown_r"], 2) if out.get("max_drawdown_r") else None
    return out


def label(how) -> str:
    return f"fixed 1:{how:.1f}" if isinstance(how, float) else f"dynamic {how}"


def main() -> int:
    symbols = {s: load_symbol(imp.RAW / s, imp.CACHE) for s in TRAINING_SYMBOLS}
    for p in imp.NEW_PAIRS:
        symbols[p] = load_symbol(imp.RAW / p)
    frames = build_all(symbols, session_start_hour=1)
    names = list(frames["EURJPY"].names)
    ej = DynamicSeries(imp.Series("EURJPY", symbols, frames, DEV_END), names)
    sets = {COMPARISON: (ej, (DEV_END, imp.SELECTION_END)), "EURJPY 2025-09 -> 2026-09": (ej, (imp.SELECTION_END, 2**62))}
    for p in imp.NEW_PAIRS:
        start = int(frames[p].close_time[imp.FULL_WINDOW_BARS])
        sets[f"{p} 2026-04 -> 2026-09 (unseen pair)"] = (DynamicSeries(imp.Series(p, symbols, frames, start), names), (0, 2**62))

    hows = [*FIXED, *DYNAMIC]
    results: dict = {}
    for set_name, (series, window) in sets.items():
        results[set_name] = {}
        for rule_name, rule in RULES.items():
            for family in FAMILIES:
                block = {"reference 1:1": rrmod.stats(series.s.run({**rule, "stop_atr": 2.0, "target_rr": 1.0}, window), series.s.pair, 1.0)}
                for how in hows:
                    tr, rr_used = series.run(rule, family, how, window)
                    block[label(how)] = summarize(tr, series.s.pair, rr_used)
                results[set_name][f"{rule_name} | {family}"] = block
        print("done", set_name)

    # choose on the comparison set only
    rng = np.random.default_rng(3)
    chosen = {}
    for key, block in results[COMPARISON].items():
        rows = {k: v for k, v in block.items() if k != "reference 1:1" and v.get("trades", 0) >= MIN_TRADES and v.get("ret_over_dd") is not None}
        best_fixed = max((k for k in rows if k.startswith("fixed")), key=lambda k: rows[k]["ret_over_dd"])
        best_dynamic = max((k for k in rows if k.startswith("dynamic")), key=lambda k: rows[k]["ret_over_dd"])
        chosen[key] = {"best_fixed": best_fixed, "best_dynamic": best_dynamic}

    checks: dict = {}
    for set_name, (series, window) in sets.items():
        checks[set_name] = {}
        for key, pick in chosen.items():
            rule_name, family = key.split(" | ")
            rule = RULES[rule_name]
            item = {"reference 1:1": results[set_name][key]["reference 1:1"]}
            for tag, name in pick.items():
                how = float(name.split("1:")[1]) if name.startswith("fixed") else name.split(" ", 1)[1]
                s = dict(results[set_name][key][name])
                d = series.s.direction(rule["keep"])
                rand = []
                for _ in range(RANDOM_RUNS):
                    signs = np.where(rng.random(len(d)) < 0.5, 1, -1) * (d != 0)
                    rt, _ = series.run({**rule, "side": "both"}, family, how, window, direction=signs)
                    rand.append(float(rt.r_net.mean()) if len(rt) else np.nan)
                s["random_runs_beating"] = round(float(np.mean(np.array(rand) >= s.get("avg_r", 0))), 3)
                item[f"{tag}: {name}"] = s
            checks[set_name][key] = item

    report = {"fixed_ratios": FIXED, "dynamic_rules": DYNAMIC, "families": {"far_target": "stop 2 ATR, target RR x stop",
              "close_stop": "target 2 ATR, stop 2 ATR / RR"}, "chosen_on": COMPARISON, "chosen": chosen, "checks": checks, "scan": results}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ML_EDGE_JPY_RR_DYNAMIC.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_txt(report, RESULTS / "ML_EDGE_JPY_RR_DYNAMIC.txt")
    for set_name, blocks in checks.items():
        print(set_name)
        for key, item in blocks.items():
            print("  ", key)
            for k, s in item.items():
                print(f"      {k:<40} n {s['trades']:>4} win {s['win_rate']:.2f} lose {s['losing_trades']:>4} avgR {s['avg_r']:+.3f} "
                      f"net {s['net_r']:+7.1f} DD {s['max_drawdown_r']} streak {s['longest_losing_streak']} "
                      f"R@1.5pip {s['avg_r_spread_sens_high']:+.3f} rand {s.get('random_runs_beating', '-')}")
    return 0


def row(name: str, s: dict) -> str:
    if not s.get("trades"):
        return f"    {name:<34} no trades"
    return (f"    {name:<34}{s['trades']:>7}{s.get('avg_rr', 1.0):>7.2f}{100 * s['win_rate']:>7.1f}{100 * s['breakeven_win_rate_needed']:>7.1f}"
            f"{s['losing_trades']:>8}{s['avg_r']:>+8.3f}{(s['profit_factor'] or 0):>7.2f}{s['net_r']:>+8.1f}{s['max_drawdown_r']:>7}"
            f"{s['longest_losing_streak']:>7}{s['exit_tp']:>5}{s['exit_sl']:>5}{s['exit_time_12h'] + s['exit_time_friday']:>6}"
            f"{s['avg_r_spread_sens_mid']:>+8.3f}{s['avg_r_spread_sens_high']:>+8.3f}"
            + (f"{100 * s['random_runs_beating']:>7.0f}" if "random_runs_beating" in s else ""))


def write_txt(report: dict, path: Path) -> None:
    head = (f"    {'case':<34}{'trades':>7}{'avgRR':>7}{'win%':>7}{'need%':>7}{'losers':>8}{'avgR':>8}{'PF':>7}{'netR':>8}{'maxDD':>7}"
            f"{'streak':>7}{'TP':>5}{'SL':>5}{'time':>6}{'R@0.5p':>8}{'R@1.5p':>8}")
    L = ["EDGE HUNTER - B1 EURJPY model on yen pairs: R:R 1:4.1 -> 1:6 (fixed scan, step 0.1) and dynamic per-trade R:R (research only)",
         "=" * 140,
         "far_target : stop 2 x ATR14(M15), target RR x stop.   close_stop : target 2 x ATR, stop 2 x ATR / RR.",
         "dynamic RR = 4.1 + 1.9 x p, p = percentile of |forecast| (strength) or vol_level_z (volatility) among the previous 60 days; _inv = 1 - p.",
         "Entry next minute open after the 15-minute decision bar; 12 h cap; Friday 20:45 UTC exit; spread 0.1 pip; swap 0.5 pip/night (Wed x3).",
         "need% = break-even win rate 1/(1+RR).  time = exits by 12 h or Friday.  R@0.5p / R@1.5p = same trades re-costed with a 0.5 / 1.5 pip spread.",
         f"Choice made on {report['chosen_on']} only (net R / max DD, >= {MIN_TRADES} trades); rand% = random-direction runs >= the rule.", ""]
    L.append("PART 1 - CHOSEN RULES CHECKED ON EVERY SET")
    for set_name, blocks in report["checks"].items():
        L.append(set_name)
        for key, item in blocks.items():
            L.append(f"  {key}")
            L.append(head + f"{'rand%':>7}")
            for k, s in item.items():
                L.append(row(k, s))
        L.append("")
    L.append("PART 2 - FULL SCAN")
    for set_name, blocks in report["scan"].items():
        L.append(set_name)
        for key, block in blocks.items():
            L.append(f"  {key}")
            L.append(head)
            for k, s in block.items():
                L.append(row(k, s))
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

"""Test a new pair with the saved EDGE ML models, without training anything on it.

The pair is unseen by every model. Applied unchanged:
  - the GLOBAL model (trained on all six symbols; meant to generalise across symbols)
  - the EURJPY model (the most consistent one; does "the yen pattern" transfer?)
for variant A (intraday) and B1 (hold to TP/SL, 12 h cap, no weekend, swap).
The self-calibrating threshold, costs (0.1 pip spread, 0.5 pip swap/night) and execution
are the research ones. Features start only once every rolling window (60 trading days of
M15) is full, then the threshold needs its 20-day warm-up.

    python scripts/ml_edge_new_pair.py AUDJPY
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_edge.data import TRAINING_SYMBOLS, cost_price, load_symbol, swap_price  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.live import ACTIVITY_FEATURES  # noqa: E402
from app.ml_edge.model import SymbolModel  # noqa: E402
from app.ml_edge.walkforward import execute, execution_config  # noqa: E402
from app.research.intraday import DAY, drift_benchmark, full_metrics, make_orders, simulate  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
RESULTS = PROJECT_ROOT / "docs" / "results"
BASE = TRAINING_SYMBOLS
VARIANTS = {
    "A": {"dir": PROJECT_ROOT / "data" / "models" / "edge_ml", "session_start": 7,
          "title": "A - intraday (exit by 4 h / 20:45 UTC)"},
    "B1": {"dir": PROJECT_ROOT / "data" / "models" / "edge_ml_v2" / "B1", "session_start": 1,
           "title": "B1 - hold to TP/SL, 12 h cap, Friday 20:45 exit, swap"},
}
FULL_WINDOW_BARS = 96 * 60  # longest rolling feature window (vol_level_z), in M15 bars
RANDOM_RUNS = 200
KEYS = ("trades", "win_rate", "avg_r", "median_r", "profit_factor", "net_r", "avg_r_ci95", "max_drawdown_r")


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def metrics(trades) -> dict:
    m = full_metrics(trades)
    return {k: m.get(k) for k in KEYS}


def pair_model(saved: SymbolModel, pair: str, architecture: str) -> SymbolModel:
    return replace(saved, symbol=pair, architecture=architecture, enabled=True, cost_price=cost_price(pair),
                   swap_per_night=swap_price(pair) if saved.label.hold else 0.0)


def orders(times, direction, atr, label):
    sel = direction != 0
    n = int(sel.sum())
    return make_orders(times[sel], direction[sel], np.full(n, np.nan), np.full(n, np.nan),
                       stop_distance=label.barrier_atr * atr[sel], target_rr=np.ones(n))


def monthly(trades) -> dict:
    out = {}
    months = trades.entry_time.astype("datetime64[s]").astype("datetime64[M]").astype(str)
    for m in sorted(set(months.tolist())):
        r = trades.r_net[months == m]
        out[m] = {"trades": int(len(r)), "avg_r": round(float(r.mean()), 3), "net_r": round(float(r.sum()), 2)}
    return out


def exit_reason(outcome: int, exit_t: int, hold: bool) -> str:
    if outcome == 1:
        return "TP"
    if outcome == -1:
        return "SL"
    sod = exit_t % DAY
    if hold:
        return "Friday" if ((exit_t // DAY + 3) % 7 == 4 and sod >= 20 * 3600 + 44 * 60) else "12h"
    return "session" if sod >= 20 * 3600 + 44 * 60 else "4h"


def main(pair: str) -> int:
    symbols = {s: load_symbol(RAW / s, CACHE) for s in BASE}
    symbols[pair] = load_symbol(RAW / pair)
    data = symbols[pair]
    report: dict = {"pair": pair, "data_utc": [iso(data.m1.open_time[0]), iso(data.m1.open_time[-1])],
                    "m1_bars": int(len(data.m1)), "costs": {"spread": cost_price(pair), "swap_per_night_B1": swap_price(pair)},
                    "variants": {}}
    trade_log = []
    rng = np.random.default_rng(11)
    for name, cfg in VARIANTS.items():
        frames = build_all(symbols, session_start_hour=cfg["session_start"])
        frame = frames[pair]
        start = int(frame.close_time[min(FULL_WINDOW_BARS, len(frame.close_time) - 1)])
        rows = frame.tradable & (frame.close_time >= start)
        times, x, atr = frame.close_time[rows], frame.x[rows], frame.atr[rows]
        x_live = x.copy()
        for f in ACTIVITY_FEATURES:
            x_live[:, frame.names.index(f)] = 0.0
        candidates = {
            "global": pair_model(SymbolModel.load(cfg["dir"] / "AUDUSD"), pair, "global"),
            "EURJPY_model": pair_model(SymbolModel.load(cfg["dir"] / "EURJPY"), pair, "symbol"),
        }
        out = {"title": cfg["title"], "decisions_from": iso(start), "models": {}}
        for mname, model in candidates.items():
            label = model.label
            direction, forecast = model.decide(times, x, atr)
            od = orders(times, direction, atr, label)
            tr = execute(data, od, label)
            item = {"keep": model.keep, "all": metrics(tr), "monthly": monthly(tr)}
            first_trade = iso(tr.entry_time[0]) if len(tr) else None
            item["first_trade"] = first_trade
            # live mode (no tick volume / spread in the live feed)
            d_live, _ = model.decide(times, x_live, atr)
            item["live_mode_no_volume_spread"] = metrics(execute(data, orders(times, d_live, atr, label), label))
            # benchmarks at the same fills: reversed, always long, always short, random direction
            item["reversed"] = metrics(execute(data, orders(times, -direction, atr, label), label))
            cfg_exec = execution_config(pair, label)
            item["always_long_same_times"] = metrics(simulate(data.m1, drift_benchmark(tr, od, 1, data.m1), cfg_exec))
            item["always_short_same_times"] = metrics(simulate(data.m1, drift_benchmark(tr, od, -1, data.m1), cfg_exec))
            rand = []
            for _ in range(RANDOM_RUNS):
                signs = np.where(rng.random(len(times)) < 0.5, 1, -1) * (direction != 0)
                rt = execute(data, orders(times, signs, atr, label), label)
                rand.append(float(rt.r_net.mean()) if len(rt) else np.nan)
            rand = np.array(rand)
            real = item["all"]["avg_r"] or 0.0
            item["random_direction"] = {"runs": RANDOM_RUNS, "mean_avg_r": round(float(np.nanmean(rand)), 4),
                                        "p95_avg_r": round(float(np.nanpercentile(rand, 95)), 4),
                                        "share_of_random_runs_beating_model": round(float(np.mean(rand >= real)), 3)}
            out["models"][mname] = item
            print(name, mname, json.dumps({k: item[k] for k in ("all", "live_mode_no_volume_spread", "reversed", "random_direction")}))
            pred = dict(zip(times.tolist(), forecast.tolist()))
            for i in range(len(tr)):
                risk = abs(tr.entry[i] - tr.stop[i])
                nights = int(tr.swap_nights[i]) if tr.swap_nights is not None else 0
                trade_log.append({
                    "variant": name, "model": mname, "signal": int(tr.signal_time[i]), "entry_time": int(tr.entry_time[i]),
                    "exit_time": int(tr.exit_time[i]), "dir": "BUY" if tr.direction[i] > 0 else "SELL",
                    "entry": float(tr.entry[i]), "sl": float(tr.stop[i]), "tp": float(tr.entry[i] + tr.direction[i] * risk),
                    "exit": float(tr.exit[i]), "reason": exit_reason(int(tr.outcome[i]), int(tr.exit_time[i]), label.hold),
                    "r_gross": float(tr.r_gross[i]), "cost_r": float((cost_price(pair) + model.swap_per_night * nights) / risk),
                    "nights": nights, "r_net": float(tr.r_net[i]), "forecast": float(pred.get(int(tr.signal_time[i]), np.nan)),
                })
        # context: the same models on EURJPY itself over the same months
        ej = frames["EURJPY"]
        erows = ej.tradable & (ej.close_time >= start)
        emodel = SymbolModel.load(cfg["dir"] / "EURJPY")
        ed, _ = emodel.decide(ej.close_time[erows], ej.x[erows], ej.atr[erows])
        out["context_EURJPY_same_months"] = metrics(execute(symbols["EURJPY"], orders(ej.close_time[erows], ed, ej.atr[erows], emodel.label), emodel.label))
        report["variants"][name] = out

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"NEW_PAIR_{pair}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_txt(pair, report, trade_log, RESULTS / f"NEW_PAIR_{pair}_TRADES.txt")
    print("wrote", RESULTS / f"NEW_PAIR_{pair}.json")
    return 0


def write_txt(pair: str, report: dict, log: list[dict], path: Path) -> None:
    L = []
    w = L.append
    w(f"EDGE HUNTER - {pair}: saved EDGE ML models applied to an UNSEEN pair (research only; not used by the app)")
    w("=" * 120)
    w(f"Data: M1 {report['data_utc'][0]} -> {report['data_utc'][1]} UTC ({report['m1_bars']} bars), MetaQuotes-Demo, Europe/Athens -> UTC.")
    w("Models: trained 2020-01 -> 2023-11-14 on the six original symbols; NOTHING was trained or tuned on this pair.")
    w("  global       = the cross-symbol model;  EURJPY_model = the EURJPY per-symbol model applied to this pair.")
    w("Trade: market entry at the next minute open after the 15-minute decision bar; SL = TP = 2 x ATR14(M15) -> R:R 1:1; stop-first.")
    w("       A: exit by 4 h or 20:45 UTC.  B1: exit after 12 h or Friday 20:45 UTC, swap charged.")
    w(f"Costs: spread {report['costs']['spread']} (0.1 pip); B1 swap {report['costs']['swap_per_night_B1']} per night (0.5 pip), Wednesday x3.")
    w("R_gross = price move / SL distance; cost_R = (spread + swap x nights) / SL distance; R_net = R_gross - cost_R.")
    w("")
    for name, v in report["variants"].items():
        w("#" * 120)
        w(f"VARIANT {v['title']}   decisions from {v['decisions_from']} (after the 60-day feature windows are full)")
        for mname, it in v["models"].items():
            a = it["all"]
            w(f"  {mname:<13} keep {it['keep']}: trades {a['trades']}  win {100 * (a['win_rate'] or 0):.1f}%  avg R {a['avg_r']:+.3f}  "
              f"median {a['median_r']:+.3f}  PF {a['profit_factor']}  net {a['net_r']:+.1f}R  95% CI of avg R {a['avg_r_ci95']}  max DD {a['max_drawdown_r']}R")
            for label, key in (("live mode (no volume/spread)", "live_mode_no_volume_spread"), ("reversed direction", "reversed"),
                               ("always BUY same times", "always_long_same_times"), ("always SELL same times", "always_short_same_times")):
                m = it[key]
                w(f"      {label:<30} trades {m['trades']}  avg R {m['avg_r']:+.3f}  PF {m['profit_factor']}")
            rd = it["random_direction"]
            w(f"      random direction ({rd['runs']} runs): mean avg R {rd['mean_avg_r']:+.3f}, 95th pct {rd['p95_avg_r']:+.3f}; "
              f"{100 * rd['share_of_random_runs_beating_model']:.0f}% of random runs >= model")
            w("      by month: " + "  ".join(f"{k} {m['trades']}t {m['avg_r']:+.2f}" for k, m in it["monthly"].items()))
        c = v["context_EURJPY_same_months"]
        w(f"  context - EURJPY model on EURJPY, same months: trades {c['trades']}  avg R {c['avg_r']:+.3f}  PF {c['profit_factor']}")
        w("")
    w("=" * 120)
    w("TRADE LIST")
    cols = (f"{'#':>4} {'signal (UTC)':<16} {'entry (UTC)':<16} {'dir':<4} {'entry':>9} {'SL':>9} {'TP':>9} {'R:R':>4} {'exit (UTC)':<16} {'exit':>9} "
            f"{'reason':<7} {'min':>5} {'R_gross':>8} {'cost_R':>7} {'nights':>6} {'R_net':>7} {'cum_R':>7}")
    for name in report["variants"]:
        for mname in report["variants"][name]["models"]:
            recs = sorted((r for r in log if r["variant"] == name and r["model"] == mname), key=lambda r: r["entry_time"])
            w("")
            w(f"--- {name} | {mname} | {len(recs)} trades")
            w(cols)
            cum = 0.0
            for k, r in enumerate(recs, 1):
                cum += r["r_net"]
                w(f"{k:>4} {iso(r['signal']):<16} {iso(r['entry_time']):<16} {r['dir']:<4} {r['entry']:>9.3f} {r['sl']:>9.3f} {r['tp']:>9.3f} {'1:1':>4} "
                  f"{iso(r['exit_time']):<16} {r['exit']:>9.3f} {r['reason']:<7} {(r['exit_time'] - r['entry_time']) // 60:>5} {r['r_gross']:>+8.3f} "
                  f"{r['cost_r']:>7.3f} {r['nights']:>6} {r['r_net']:>+7.3f} {cum:>+7.2f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1].upper() if len(sys.argv) > 1 else "AUDJPY"))

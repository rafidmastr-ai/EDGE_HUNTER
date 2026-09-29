"""EDGE ML study: learn from all symbols, specialise per symbol, test out of sample.

Research only: models are written to data/models/edge_ml/<SYMBOL>/ (git-ignored) and
are NOT used by the app. Report: docs/results/ML_EDGE_REPORT.md (+ .json, manifests).

    python scripts/ml_edge_research.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_edge.data import discover_symbols, load_symbol  # noqa: E402
from app.ml_edge.features import FEATURE_VERSION, build_all  # noqa: E402
from app.ml_edge.model import HYPER, SymbolModel, edges  # noqa: E402
from app.ml_edge.walkforward import (  # noqa: E402
    DEV_END,
    FOLDS,
    LABELS,
    OOS_END,
    build_rows,
    choose_symbol,
    execute,
    final_models,
    fit_models,
    forecast,
    orders_from,
    spearman,
    stage1,
    validation_forecasts,
    validation_mask,
)
from app.research.intraday import ExecutionConfig, drift_benchmark, full_metrics, make_orders, simulate  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
MODELS = PROJECT_ROOT / "data" / "models" / "edge_ml"
RESULTS = PROJECT_ROOT / "docs" / "results"
SEGMENTS = {"validation": (FOLDS[0][0], DEV_END), "oos": (DEV_END, OOS_END), "forward_2025": (OOS_END, None)}


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def seg_trades(trades, name):
    start, end = SEGMENTS[name]
    m = trades.entry_time >= start
    if end is not None:
        m &= trades.entry_time < end
    return trades.select(m)


def main() -> int:
    t_start = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - t_start:6.0f}s] {msg}", flush=True)  # noqa: E731
    symbols = {s: load_symbol(RAW / s, CACHE) for s in discover_symbols(RAW)}
    frames = build_all(symbols)
    feature_names = next(iter(frames.values())).names
    rows = build_rows(symbols, frames)
    log(f"data: {', '.join(f'{s}={len(r.time)}' for s, r in rows.items())} rows, {len(feature_names)} features")
    report: dict = {"symbols": list(rows), "features": list(feature_names), "feature_version": FEATURE_VERSION,
                    "folds": [[iso(a), iso(b)] for a, b in FOLDS], "dev_end": iso(DEV_END), "oos_end": iso(OOS_END)}

    # ---- stage 1: label design + hyper-parameters (pooled, validation IC of the global model)
    table = stage1(rows, HYPER, progress=log)
    (label_key, hyper) = max(table, key=lambda k: table[k]["mean_ic"])
    label = next(l for l in LABELS if l.key == label_key)
    report["stage1"] = {f"{k[0]}|{k[1]}": v for k, v in table.items()}
    report["stage1_choice"] = {"label": label_key, "hyper": hyper}
    log(f"stage 1 choice: {label_key} {hyper}")

    # ---- stage 2: architecture + threshold per symbol (validation folds)
    fc = validation_forecasts(rows, label, hyper)
    choices = {s: choose_symbol(rows[s], label, fc[s], symbols_cost(s)) for s in rows}
    report["choices"] = {s: c["chosen"] for s, c in choices.items()}
    trials = len(table) + sum(c["trials"] for c in choices.values())
    report["trials"] = {"stage1_label_x_hyper": len(table), "stage2_arch_x_threshold_per_symbol": {s: c["trials"] for s, c in choices.items()}, "total": trials}
    for s, c in choices.items():
        ch = c["chosen"]
        log(f"  {s}: {ch['architecture']} keep={ch['keep']} theta={ch['theta']:+.4f} val net R={ch['mean_net_r']:+.4f} folds+={ch['positive_folds']}/3 enabled={ch['enabled']}")

    # validation trades (harness) with the chosen architecture / threshold per symbol
    results: dict = {s: {} for s in rows}
    for s, r in rows.items():
        ch = choices[s]["chosen"]
        vm = validation_mask(r)
        _, f = fc[s][ch["architecture"]]
        cost_r = symbols_cost(s) / (label.barrier_atr * r.atr[vm])
        edge, direction = edges(f, cost_r)
        full_dir = np.zeros(len(r.time), int)
        full_dir[np.flatnonzero(vm)] = np.where(edge > ch["theta"], direction, 0)
        trades = execute(symbols[s], orders_from(r, full_dir, label, vm), label)
        results[s]["validation"] = full_metrics(trades)
        results[s]["validation_by_fold"] = {
            f"{iso(a)[:10]}..{iso(b)[:10]}": full_metrics(trades.select((trades.entry_time >= a) & (trades.entry_time < b))).get("avg_r")
            for a, b in FOLDS
        }

    # ---- final fit on the whole development period; OOS + forward, run once
    final = final_models(rows, label, hyper, choices)
    MODELS.mkdir(parents=True, exist_ok=True)
    manifests_dir = RESULTS / "edge_ml"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    spot: dict = {}
    for s, r in rows.items():
        model = final[s]
        model.feature_names = feature_names
        model.info = {"trained_through": iso(DEV_END), "validation": choices[s]["chosen"], "stage1": report["stage1_choice"]}
        after = r.mask(DEV_END, None)
        direction, fcast = model.decide(r.x[after], r.atr[after])
        shadow = SymbolModel(**{**model.__dict__, "enabled": True})
        shadow_dir, _ = shadow.decide(r.x[after], r.atr[after])
        full_dir = np.zeros(len(r.time), int)
        full_shadow = np.zeros(len(r.time), int)
        full_dir[np.flatnonzero(after)] = direction
        full_shadow[np.flatnonzero(after)] = shadow_dir
        orders = orders_from(r, full_shadow, label, after)
        by_level = {level: execute(symbols[s], orders, label, level) for level in ("base", "sens_mid", "sens_high")}
        for seg in ("oos", "forward_2025"):
            base = seg_trades(by_level["base"], seg)
            entry = {"model_enabled": model.enabled, **full_metrics(base)}
            entry["avg_r_sens_mid"] = full_metrics(seg_trades(by_level["sens_mid"], seg)).get("avg_r")
            entry["avg_r_sens_high"] = full_metrics(seg_trades(by_level["sens_high"], seg)).get("avg_r")
            if len(base):
                bench = {}
                for lab, d in (("always_long", 1), ("always_short", -1)):
                    b = simulate(symbols[s].m1, drift_benchmark(base, orders, d, symbols[s].m1),
                                 ExecutionConfig(cost=symbols_cost(s), horizon_bars=label.horizon_bars * 15, one_position=False))
                    bench[lab] = full_metrics(b).get("avg_r")
                rnd = []
                for seed in range(20):
                    rng = random.Random(seed)
                    dirs = np.array([rng.choice((-1, 1)) for _ in range(len(base))])
                    ro = make_orders(base.signal_time, dirs, np.full(len(base), np.nan), np.full(len(base), np.nan),
                                     stop_distance=np.abs(base.entry - base.stop), target_rr=np.ones(len(base)))
                    rt = simulate(symbols[s].m1, ro, ExecutionConfig(cost=symbols_cost(s), horizon_bars=label.horizon_bars * 15, one_position=False))
                    rnd.append(rt.r_net.mean() if len(rt) else np.nan)
                bench["random_direction_mean_of_20"] = round(float(np.nanmean(rnd)), 4)
                entry["benchmarks"] = bench
            results[s][seg] = entry
        # reproducibility: saved model must reproduce the same decisions
        model.save(MODELS / s)
        loaded = SymbolModel.load(MODELS / s)
        again, _ = loaded.decide(r.x[after], r.atr[after])
        results[s]["reload_identical_decisions"] = bool(np.array_equal(again, direction))
        shutil.copy(MODELS / s / "manifest.json", manifests_dir / f"{s}_manifest.json")
        oos_tr = seg_trades(by_level["base"], "oos")
        pick = sorted(random.Random(1).sample(range(len(oos_tr)), min(5, len(oos_tr))))
        index = {int(t): i for i, t in enumerate(r.time)}
        spot[s] = [{
            "signal": iso(oos_tr.signal_time[i]), "direction": "BUY" if oos_tr.direction[i] > 0 else "SELL",
            "forecast_gross_r": round(float(shadow.forecast(r.x[[index[int(oos_tr.signal_time[i])]]])[0]), 4),
            "theta": round(model.theta, 4), "entry": round(float(oos_tr.entry[i]), 5), "stop": round(float(oos_tr.stop[i]), 5),
            "exit_time": iso(oos_tr.exit_time[i]), "exit": round(float(oos_tr.exit[i]), 5), "r_net": round(float(oos_tr.r_net[i]), 3),
        } for i in pick]
        log(f"  {s}: OOS avg R={results[s]['oos'].get('avg_r')} n={results[s]['oos'].get('trades')} | 2025 avg R={results[s]['forward_2025'].get('avg_r')} | enabled={model.enabled}")
    report["results"] = results
    report["spot_check"] = spot

    # ---- feature influence (fold 3 validation, global model): IC drop when a feature is shuffled
    models3 = fit_models(rows, label.key, hyper, FOLDS[2][0], archs=("global",))
    base_ic, drops = [], {n: [] for n in feature_names}
    rng = np.random.default_rng(0)
    for s, r in rows.items():
        m = r.mask(*FOLDS[2])
        x, y = r.x[m], r.y[label.key][m]
        ic = spearman(forecast(models3, "global", s, x), y)
        base_ic.append(ic)
        for j, n in enumerate(feature_names):
            xs = x.copy()
            xs[:, j] = rng.permutation(xs[:, j])
            drops[n].append(ic - spearman(forecast(models3, "global", s, xs), y))
    report["feature_influence_fold3"] = {"base_ic_mean": round(float(np.nanmean(base_ic)), 4),
                                         "ic_drop": dict(sorted(((n, round(float(np.nanmean(v)), 5)) for n, v in drops.items()), key=lambda kv: -kv[1]))}
    log("feature influence done")

    # ---- null test: training labels circularly shifted (real features, meaningless targets)
    shift = {s: random.Random(i).uniform(0.3, 0.7) for i, s in enumerate(rows)}
    fc_null = validation_forecasts(rows, label, hyper, label_shift=shift)
    choices_null = {s: choose_symbol(rows[s], label, fc_null[s], symbols_cost(s)) for s in rows}
    final_null = final_models(rows, label, hyper, choices_null, label_shift=shift)
    null = {}
    for s, r in rows.items():
        after = r.mask(DEV_END, None)
        shadow = SymbolModel(**{**final_null[s].__dict__, "enabled": True})
        d, _ = shadow.decide(r.x[after], r.atr[after])
        full = np.zeros(len(r.time), int)
        full[np.flatnonzero(after)] = d
        tr = execute(symbols[s], orders_from(r, full, label, after), label)
        null[s] = {"validation_net_r": round(choices_null[s]["chosen"]["mean_net_r"], 4), "enabled": choices_null[s]["chosen"]["enabled"],
                   "oos_avg_r": full_metrics(seg_trades(tr, "oos")).get("avg_r"), "forward_avg_r": full_metrics(seg_trades(tr, "forward_2025")).get("avg_r")}
    report["null_test_shifted_labels"] = null
    log("null test done")

    # ---- acceptance per symbol (rules fixed in the plan)
    verdict = {}
    for s in rows:
        oos, fwd = results[s]["oos"], results[s]["forward_2025"]
        bench = oos.get("benchmarks") or {}
        long_share = oos.get("long_trades", 0) / max(1, oos.get("trades", 1))
        drift = bench.get("always_long") if long_share >= 0.5 else bench.get("always_short")
        ci = oos.get("avg_r_ci95") or [None, None]
        checks = {
            "enabled after validation": bool(choices[s]["chosen"]["enabled"]),
            "OOS edge (CI low > 0, or PF >= 1.15 with >= 100 trades)": bool((ci[0] is not None and ci[0] > 0) or ((oos.get("profit_factor") or 0) >= 1.15 and oos.get("trades", 0) >= 100)),
            "2025 not negative": (fwd.get("avg_r") or 0) >= 0,
            "positive at 0.5 pip / 0.30 $": (oos.get("avg_r_sens_mid") or -1) > 0,
            "beats drift benchmark": drift is not None and (oos.get("avg_r") or -1) > drift,
            "beats random direction": (oos.get("avg_r") or -1) > (bench.get("random_direction_mean_of_20") or 0),
        }
        verdict[s] = {"accepted": all(checks.values()), "checks": checks}
    report["verdict"] = verdict
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ML_EDGE_REPORT.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log(f"done: accepted = {[s for s, v in verdict.items() if v['accepted']]}")
    return 0


def symbols_cost(symbol: str) -> float:
    from app.ml_edge.data import cost_price

    return cost_price(symbol, "base")


if __name__ == "__main__":
    raise SystemExit(main())

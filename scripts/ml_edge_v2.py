"""EDGE ML v2 study on the extended data (to 2026-09-28). Research only; the app is untouched.

A   the models saved by run 2 (data/models/edge_ml/<SYM>) run unchanged on the new data;
B1  hold until TP/SL, forced close after 12 h (never over the weekend), estimated swap.
    Same protocol as v1: stage 1 label/hyper on validation IC, stage 2 architecture +
    keep fraction per symbol on validation, final fit on development, 5 null runs.

Segments: validation folds 2021..2023-11-14 | OOS 2023-11-14..2024-12-31 |
2025-01..2025-08 (seen in earlier reports) | NEW FORWARD 2025-09-01..2026-09-28 (never seen).

    python scripts/ml_edge_v2.py --variants A,B1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_edge.data import TRAINING_SYMBOLS, cost_price, discover_symbols, load_symbol, swap_price  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.model import HYPER, SymbolModel  # noqa: E402
from app.ml_edge.walkforward import (  # noqa: E402
    DEV_END,
    FOLDS,
    HOLD_LABELS,
    OOS_END,
    build_rows,
    choose_symbol,
    execute,
    execution_config,
    final_models,
    orders_from,
    stage1,
    ts,
    validation_forecasts,
    validation_mask,
)
from app.research.intraday import drift_benchmark, full_metrics, make_orders, simulate  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
MODELS_V1 = PROJECT_ROOT / "data" / "models" / "edge_ml"
MODELS_V2 = PROJECT_ROOT / "data" / "models" / "edge_ml_v2"
RESULTS = PROJECT_ROOT / "docs" / "results"
NEW_FORWARD = ts("2025-09-01")
DATA_END_REQUIRED = ts("2026-09-25")
SEGMENTS = {
    "validation": (FOLDS[0][0], DEV_END),
    "oos": (DEV_END, OOS_END),
    "fwd_2025_jan_aug": (OOS_END, NEW_FORWARD),
    "new_forward": (NEW_FORWARD, None),
}


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def seg(trades, name):
    start, end = SEGMENTS[name]
    m = trades.entry_time >= start
    if end is not None:
        m &= trades.entry_time < end
    return trades.select(m)


def check_data(symbols) -> list[str]:
    problems = []
    for s, d in symbols.items():
        t = d.m1.open_time
        if t[-1] < DATA_END_REQUIRED:
            problems.append(f"{s}: data ends {iso(t[-1])}")
        after = t[t >= ts("2025-08-25")]
        gaps = np.diff(after)
        big = np.flatnonzero(gaps > 4 * 86400)
        for i in big:
            problems.append(f"{s}: gap {iso(after[i])} -> {iso(after[i + 1])}")
    return problems


def metrics_with_benchmarks(symbol_data, trades, orders, label, benchmarks: bool) -> dict:
    out = full_metrics(trades)
    if trades.swap_nights is not None and len(trades):
        out["swap_nights_total"] = int(trades.swap_nights.sum())
    if benchmarks and len(trades):
        cfg = execution_config(symbol_data.symbol, label, one_position=False)
        bench = {}
        for lab, d in (("always_long", 1), ("always_short", -1)):
            bench[lab] = full_metrics(simulate(symbol_data.m1, drift_benchmark(trades, orders, d, symbol_data.m1), cfg)).get("avg_r")
        rnd = []
        for seed in range(20):
            rng = random.Random(seed)
            dirs = np.array([rng.choice((-1, 1)) for _ in range(len(trades))])
            ro = make_orders(trades.signal_time, dirs, np.full(len(trades), np.nan), np.full(len(trades), np.nan),
                             stop_distance=np.abs(trades.entry - trades.stop), target_rr=np.ones(len(trades)))
            rt = simulate(symbol_data.m1, ro, cfg)
            rnd.append(rt.r_net.mean() if len(rt) else np.nan)
        bench["random_direction_mean_of_20"] = round(float(np.nanmean(rnd)), 4)
        out["benchmarks"] = bench
    return out


def evaluate_models(symbols, rows, models: dict[str, SymbolModel], label, *, shadow: bool = True) -> dict:
    """Run each model on the stream after DEV_END (continuous rolling threshold) at all cost levels."""
    results = {}
    for s, r in rows.items():
        model = models[s]
        run = SymbolModel(**{**model.__dict__, "enabled": True}) if shadow else model
        after = r.mask(DEV_END, None)
        direction, _ = run.decide(r.time[after], r.x[after], r.atr[after])
        full = np.zeros(len(r.time), int)
        full[np.flatnonzero(after)] = direction
        orders = orders_from(r, full, label, after)
        levels = {
            "base": execute(symbols[s], orders, label),
            "sens_mid": execute(symbols[s], orders, label, "sens_mid"),
            "double_swap": execute(symbols[s], orders, label, swap_multiplier=2.0),
        }
        entry = {"model_enabled": model.enabled}
        for name in ("oos", "fwd_2025_jan_aug", "new_forward"):
            base = seg(levels["base"], name)
            m = metrics_with_benchmarks(symbols[s], base, orders, label, benchmarks=name in ("oos", "new_forward"))
            m["avg_r_sens_mid"] = full_metrics(seg(levels["sens_mid"], name)).get("avg_r")
            m["avg_r_double_swap"] = full_metrics(seg(levels["double_swap"], name)).get("avg_r")
            m["exit_reasons"] = {k: int((base.outcome == v).sum()) for k, v in (("target", 1), ("stop", -1), ("time_or_friday", 0))} if len(base) else {}
            entry[name] = m
        results[s] = entry
    return results


def variant_a(symbols) -> dict:
    """Saved v1 models, unchanged (intraday, feature v1 = session start 07:00)."""
    frames = build_all(symbols)
    models = {s: SymbolModel.load(MODELS_V1 / s) for s in symbols}
    label = next(iter(models.values())).label
    rows = build_rows(symbols, frames, labels=(label,))
    names = next(iter(frames.values())).names
    for s, m in models.items():
        if tuple(m.feature_names) != tuple(names):
            raise ValueError(f"{s}: saved model features differ from the current feature builder")
    return {"label": label.key, "models_from": str(MODELS_V1), "results": evaluate_models(symbols, rows, models, label)}


def variant_b(symbols, name: str, labels, session_start_hour: int, log) -> dict:
    frames = build_all(symbols, session_start_hour=session_start_hour)
    feature_names = next(iter(frames.values())).names
    rows = build_rows(symbols, frames, labels=labels)
    swap = {s: swap_price(s) for s in symbols}
    table = stage1(rows, HYPER, labels=labels, progress=log)
    label_key, hyper = max(table, key=lambda k: table[k]["mean_ic"])
    label = next(lab for lab in labels if lab.key == label_key)
    log(f"{name} stage 1: {label_key} {hyper}")
    fc = validation_forecasts(rows, label, hyper)
    choices = {s: choose_symbol(rows[s], label, fc[s], cost_price(s), swap[s]) for s in rows}
    out = {"stage1": {f"{k[0]}|{k[1]}": v for k, v in table.items()}, "stage1_choice": [label_key, hyper],
           "choices": {s: {k: v for k, v in c["chosen"].items() if not k.startswith("_")} for s, c in choices.items()},
           "trials": len(table) + sum(c["trials"] for c in choices.values()), "validation": {}}
    for s, r in rows.items():
        vm = validation_mask(r)
        full = np.zeros(len(r.time), int)
        full[np.flatnonzero(vm)] = choices[s]["chosen"]["_direction"]
        out["validation"][s] = full_metrics(execute(symbols[s], orders_from(r, full, label, vm), label))
        ch = out["choices"][s]
        log(f"  {s}: {ch['architecture']} keep={ch['keep']} val net R={ch['mean_net_r']:+.4f} enabled={ch['enabled']}")
    final = final_models(rows, label, hyper, choices, swap=swap)
    for s, m in final.items():
        m.feature_names = feature_names
        m.info = {"variant": name, "trained_through": iso(DEV_END), "validation": out["choices"][s]}
        m.save(MODELS_V2 / name / s)
        again = SymbolModel.load(MODELS_V2 / name / s)
        after = rows[s].mask(DEV_END, None)
        a, _ = m.decide(rows[s].time[after], rows[s].x[after], rows[s].atr[after])
        b, _ = again.decide(rows[s].time[after], rows[s].x[after], rows[s].atr[after])
        if not np.array_equal(a, b):
            raise RuntimeError(f"{s}: reloaded model decisions differ")
    out["results"] = evaluate_models(symbols, rows, final, label)
    null_runs = []
    for seed in range(5):
        shift = {s: random.Random(100 * seed + i).uniform(0.2, 0.8) for i, s in enumerate(rows)}
        fc_null = validation_forecasts(rows, label, hyper, label_shift=shift)
        ch_null = {s: choose_symbol(rows[s], label, fc_null[s], cost_price(s), swap[s]) for s in rows}
        fin_null = final_models(rows, label, hyper, ch_null, label_shift=shift, swap=swap)
        res_null = evaluate_models(symbols, rows, fin_null, label)
        null_runs.append({s: {"validation_net_r": round(ch_null[s]["chosen"]["mean_net_r"], 4), "enabled": ch_null[s]["chosen"]["enabled"],
                              **{k: res_null[s][k].get("avg_r") for k in ("oos", "fwd_2025_jan_aug", "new_forward")}} for s in rows})
        log(f"  {name} null run {seed + 1}/5")
    out["null_runs"] = null_runs
    return out


def verdict(entry: dict, choices: dict | None, null_runs: list | None) -> dict:
    out = {}
    for s, res in entry.items():
        oos, nf, fwd25 = res["oos"], res["new_forward"], res["fwd_2025_jan_aug"]
        bench = oos.get("benchmarks") or {}
        long_share = oos.get("long_trades", 0) / max(1, oos.get("trades", 1))
        drift = bench.get("always_long") if long_share >= 0.5 else bench.get("always_short")
        ci = oos.get("avg_r_ci95") or [None, None]
        checks = {
            "enabled after validation": bool(res["model_enabled"]),
            "OOS edge (CI low > 0, or PF >= 1.15 with >= 100 trades)": bool((ci[0] is not None and ci[0] > 0) or ((oos.get("profit_factor") or 0) >= 1.15 and oos.get("trades", 0) >= 100)),
            "2025 Jan-Aug not negative": (fwd25.get("avg_r") or 0) >= 0,
            "NEW forward 2025-09..2026-09 positive": (nf.get("avg_r") or -1) > 0,
            "positive at 0.5 pip / 0.30 $ (OOS)": (oos.get("avg_r_sens_mid") or -1) > 0,
            "beats drift benchmark (OOS)": drift is not None and (oos.get("avg_r") or -1) > drift,
        }
        if null_runs is not None and choices is not None:
            checks["validation beats all 5 null runs"] = bool(choices[s]["mean_net_r"] > max(run[s]["validation_net_r"] for run in null_runs))
        out[s] = {"accepted": all(checks.values()), "checks": checks}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="A,B1")
    args = parser.parse_args(argv)
    started = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - started:6.0f}s] {msg}", flush=True)  # noqa: E731
    symbols = {s: load_symbol(RAW / s, CACHE) for s in discover_symbols(RAW) if s in TRAINING_SYMBOLS}
    problems = check_data(symbols)
    if problems:
        print("Data incomplete — upload the missing files first:\n  " + "\n  ".join(problems))
        return 2
    report_path = RESULTS / "ML_EDGE_V2_REPORT.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["data_end"] = {s: iso(d.m1.open_time[-1]) for s, d in symbols.items()}
    report["segments"] = {k: [iso(a), iso(b) if b else "end"] for k, (a, b) in SEGMENTS.items()}
    report["costs"] = {s: {"spread": cost_price(s), "swap_per_night": swap_price(s)} for s in symbols}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if variant == "A":
            entry = variant_a(symbols)
            entry["verdict"] = verdict(entry["results"], None, None)
        elif variant == "B1":
            entry = variant_b(symbols, "B1", HOLD_LABELS, 1, log)
            entry["verdict"] = verdict(entry["results"], entry["choices"], entry["null_runs"])
        else:
            raise SystemExit(f"unknown variant {variant} (calendar variants need the calendar file)")
        report[variant] = entry
        for s, v in entry["verdict"].items():
            nf = entry["results"][s]["new_forward"]
            log(f"{variant} {s}: new forward n={nf.get('trades')} avg R={nf.get('avg_r')} PF={nf.get('profit_factor')} accepted={v['accepted']}")
        report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

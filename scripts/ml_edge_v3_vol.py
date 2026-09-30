"""Proposal 3: volatility forecast overlay on the saved A and B1 models + detailed TXT trade log.

The signal models are loaded unchanged (A: data/models/edge_ml, B1: data/models/edge_ml_v2/B1).
A volatility model is trained with the same walk-forward folds; the overlay rule per
symbol is chosen on the validation folds only (return / max drawdown), then OOS, 2025
and the new forward period are evaluated once, with and without the overlay.

    python scripts/ml_edge_v3_vol.py
"""

from __future__ import annotations

import json
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
from app.ml_edge.model import SymbolModel  # noqa: E402
from app.ml_edge.volatility import OVERLAYS, VolModel, forecast_quality, make_vol_regressor, overlay_actions, realised_vol_ratio  # noqa: E402
from app.ml_edge.walkforward import (  # noqa: E402
    DEV_END,
    EMBARGO,
    FOLDS,
    OOS_END,
    build_rows,
    choose_symbol,
    execute,
    orders_from,
    ts,
    validation_forecasts,
    validation_mask,
)
from app.research.intraday import DAY, full_metrics  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
RESULTS = PROJECT_ROOT / "docs" / "results"
VARIANTS = {
    "A": {"models": PROJECT_ROOT / "data" / "models" / "edge_ml", "session_start": 7, "title": "A - intraday (exit by 20:45 UTC / 4 h)"},
    "B1": {"models": PROJECT_ROOT / "data" / "models" / "edge_ml_v2" / "B1", "session_start": 1, "title": "B1 - hold to TP/SL, 12 h cap, no weekend, swap"},
}
NEW_FORWARD = ts("2025-09-01")
SEGMENTS = {"oos": (DEV_END, OOS_END), "fwd_2025_jan_aug": (OOS_END, NEW_FORWARD), "new_forward": (NEW_FORWARD, 2**62)}
SEGMENT_TITLES = {"oos": "OOS 2023-11-14 -> 2024-12-31", "fwd_2025_jan_aug": "2025-01 -> 2025-08", "new_forward": "NEW FORWARD 2025-09-01 -> 2026-09-28"}
CAPITAL = 10_000.0
MIN_VALIDATION_TRADES = 100


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def money(r_net: np.ndarray, risk_pct: np.ndarray, exit_time: np.ndarray) -> dict:
    if not len(r_net):
        return {"trades": 0}
    order = np.argsort(exit_time, kind="stable")
    pnl = r_net[order] * risk_pct[order] / 100.0 * CAPITAL
    equity = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[CAPITAL], equity]))[1:]
    dd = float(np.max(peak - equity)) if len(equity) else 0.0
    net = float(pnl.sum())
    return {"net_profit_usd": round(net, 2), "net_return_pct": round(net / CAPITAL * 100, 2), "max_dd_usd": round(dd, 2),
            "max_dd_pct": round(dd / CAPITAL * 100, 2), "return_over_dd": round(net / dd, 2) if dd > 0 else None,
            "avg_risk_pct": round(float(risk_pct.mean()), 3)}


def summary(trades, take: np.ndarray, risk: np.ndarray) -> dict:
    sel = trades.select(take)
    out = full_metrics(sel)
    out.update(money(sel.r_net, risk[take], sel.exit_time))
    out["skipped_by_overlay"] = int((~take).sum())
    if sel.swap_nights is not None and len(sel):
        out["swap_nights"] = int(sel.swap_nights.sum())
    return out


def exit_reason(outcome: int, exit_t: int, hold: bool) -> str:
    if outcome == 1:
        return "TP"
    if outcome == -1:
        return "SL"
    sod = exit_t % DAY
    if hold:
        return "Friday" if ((exit_t // DAY + 3) % 7 == 4 and sod >= 20 * 3600 + 45 * 60 - 60) else "12h"
    return "session" if sod >= 20 * 3600 + 44 * 60 else "4h"


def saved_validation_direction(r, label, fc_symbol: dict, model: SymbolModel) -> np.ndarray:
    """Validation decisions of the saved model's architecture and keep fraction (per-fold rolling threshold)."""
    from app.ml_edge.model import edges, expected_nights, rolling_thresholds

    vm = validation_mask(r)
    times = r.time[vm]
    folds, f = fc_symbol[model.architecture]
    cost_r = (model.cost_price + model.swap_per_night * expected_nights(times, label)) / (label.barrier_atr * r.atr[vm])
    edge, direction = edges(f, cost_r)
    threshold = np.full(len(edge), np.inf)
    for k in range(len(FOLDS)):
        fm = folds == k
        threshold[fm] = rolling_thresholds(times[fm], edge[fm], model.keep)
    return np.where(np.isfinite(r.y[label.key][vm]) & (edge > np.maximum(threshold, 0.0)), direction, 0)


def trade_overlay(trades, times: np.ndarray, take_rows: np.ndarray, risk_rows: np.ndarray):
    index = np.searchsorted(times, trades.signal_time)
    return take_rows[index], risk_rows[index]


def run_variant(name: str, cfg: dict, symbols: dict, log) -> dict:
    models = {s: SymbolModel.load(cfg["models"] / s) for s in symbols}
    label, hyper = next(iter(models.values())).label, next(iter(models.values())).hyper
    frames = build_all(symbols, session_start_hour=cfg["session_start"])
    rows = build_rows(symbols, frames, labels=(label,))
    target = {s: np.log(realised_vol_ratio(symbols[s].m1, r.time, r.atr, label)) for s, r in rows.items()}
    hour = {s: (r.time % DAY) // 3600 for s, r in rows.items()}

    # ---- volatility model: walk-forward on the development period, then final fit
    val_vol = {s: np.full(len(r.time), np.nan) for s, r in rows.items()}
    quality = []
    for train_end, val_end in FOLDS:
        tm = {s: r.mask(None, train_end - EMBARGO) & np.isfinite(target[s]) for s, r in rows.items()}
        reg = make_vol_regressor().fit(np.vstack([rows[s].x[tm[s]] for s in rows]), np.concatenate([target[s][tm[s]] for s in rows]))
        ys, ps, hs = [], [], []
        for s, r in rows.items():
            vm = r.mask(train_end, val_end)
            val_vol[s][vm] = reg.predict(r.x[vm])
            ys.append(target[s][vm]); ps.append(val_vol[s][vm]); hs.append(hour[s][vm])
        quality.append({"fold": f"{iso(train_end)[:10]}..{iso(val_end)[:10]}", **forecast_quality(
            np.concatenate(ys), np.concatenate(ps), np.concatenate(hs),
            np.concatenate([target[s][tm[s]] for s in rows]), np.concatenate([hour[s][tm[s]] for s in rows]))})
    tm = {s: r.mask(None, DEV_END - EMBARGO) & np.isfinite(target[s]) for s, r in rows.items()}
    vol_model = VolModel(label, next(iter(frames.values())).names, make_vol_regressor().fit(
        np.vstack([rows[s].x[tm[s]] for s in rows]), np.concatenate([target[s][tm[s]] for s in rows])))
    vol_model.save(cfg["models"].parent / f"vol_{name}")
    after_q = []
    for s, r in rows.items():
        am = r.mask(DEV_END, None)
        after_q.append((target[s][am], np.log(vol_model.forecast_ratio(r.x[am])), hour[s][am]))
    quality.append({"fold": "final model on OOS + 2025 + new forward", **forecast_quality(
        np.concatenate([a for a, _, _ in after_q]), np.concatenate([b for _, b, _ in after_q]), np.concatenate([c for _, _, c in after_q]),
        np.concatenate([target[s][tm[s]] for s in rows]), np.concatenate([hour[s][tm[s]] for s in rows]))})
    for q in quality:
        log(f"{name} vol quality {q['fold']}: R2 model {q['r2_model']} | hour-of-day {q['r2_hour_of_day']} | constant {q['r2_constant']}")

    # ---- validation: rebuild the saved models' validation decisions, choose the overlay per symbol
    fc = validation_forecasts(rows, label, hyper)
    out = {"label": label.key, "hyper": hyper, "vol_quality": quality, "symbols": {}}
    records = []
    for s, r in rows.items():
        model = models[s]
        reselected = choose_symbol(r, label, fc[s], cost_price(s), model.swap_per_night)["chosen"]
        vm = validation_mask(r)
        full = np.zeros(len(r.time), int)
        full[np.flatnonzero(vm)] = saved_validation_direction(r, label, fc[s], model)
        vtr = execute(symbols[s], orders_from(r, full, label, vm), label)
        rule_scores = {}
        for rule in OVERLAYS:
            take_rows = np.ones(len(r.time), bool)
            risk_rows = np.full(len(r.time), 1.0)
            for train_end, val_end in FOLDS:  # each fold's forecasts form their own series, like the signal threshold
                fm = r.mask(train_end, val_end)
                t_, k_ = overlay_actions(r.time[fm], np.exp(val_vol[s][fm]), rule)
                take_rows[fm], risk_rows[fm] = t_, k_
            take, risk = trade_overlay(vtr, r.time, take_rows, risk_rows)
            m = money(vtr.r_net[take], risk[take], vtr.exit_time[take])
            rule_scores[rule] = {"trades": int(take.sum()), **m}
        eligible = [k for k, v in rule_scores.items() if v["trades"] >= MIN_VALIDATION_TRADES and v.get("return_over_dd") is not None]
        best = max(eligible, key=lambda k: rule_scores[k]["return_over_dd"]) if eligible else "none"
        if best != "none" and rule_scores[best]["return_over_dd"] <= (rule_scores["none"].get("return_over_dd") or -1e9):
            best = "none"
        log(f"{name} {s}: enabled={model.enabled} overlay chosen on validation = {best} "
            f"(return/DD none {rule_scores['none'].get('return_over_dd')} -> {rule_scores[best].get('return_over_dd')})")

        # ---- OOS / 2025 / new forward: model unchanged (shadow run for disabled symbols), overlay applied
        after = r.mask(DEV_END, None)
        shadow = SymbolModel(**{**model.__dict__, "enabled": True})
        d, forecast_r = shadow.decide(r.time[after], r.x[after], r.atr[after])
        full = np.zeros(len(r.time), int)
        full[np.flatnonzero(after)] = d
        orders = orders_from(r, full, label, after)
        base = execute(symbols[s], orders, label)
        mid = execute(symbols[s], orders, label, "sens_mid")
        f_after = np.full(len(r.time), np.nan)
        f_after[after] = vol_model.forecast_ratio(r.x[after])
        take_rows = np.ones(len(r.time), bool)
        risk_rows = np.full(len(r.time), 1.0)
        take_rows[after], risk_rows[after] = overlay_actions(r.time[after], f_after[after], best)
        take, risk = trade_overlay(base, r.time, take_rows, risk_rows)
        pred_by_time = dict(zip(r.time[after].tolist(), forecast_r.tolist()))
        sym = {"enabled_by_validation": bool(model.enabled), "architecture": model.architecture, "keep": model.keep,
               "reselection_today": {"architecture": reselected["architecture"], "keep": reselected["keep"], "enabled": reselected["enabled"]},
               "overlay": best, "overlay_validation_scores": rule_scores, "validation_none": full_metrics(vtr), "segments": {}}
        for seg_name, (a, b) in SEGMENTS.items():
            sm = (base.entry_time >= a) & (base.entry_time < b)
            mm = (mid.entry_time >= a) & (mid.entry_time < b)
            sub = base.select(sm)
            sym["segments"][seg_name] = {
                "without_overlay": summary(sub, np.ones(len(sub), bool), np.full(len(sub), 1.0)),
                "with_overlay": summary(sub, take[sm], risk[sm]),
                "avg_r_at_0.5pip_0.30usd": full_metrics(mid.select(mm)).get("avg_r"),
            }
        out["symbols"][s] = sym
        risk_price = np.abs(base.entry - base.stop)
        swap = model.swap_per_night
        for i in range(len(base)):
            seg_name = next((k for k, (a, b) in SEGMENTS.items() if a <= base.entry_time[i] < b), None)
            if seg_name is None:
                continue
            nights = int(base.swap_nights[i]) if base.swap_nights is not None else 0
            records.append({
                "variant": name, "symbol": s, "segment": seg_name, "enabled": bool(model.enabled),
                "signal": int(base.signal_time[i]), "entry_time": int(base.entry_time[i]), "exit_time": int(base.exit_time[i]),
                "dir": "BUY" if base.direction[i] > 0 else "SELL", "entry": float(base.entry[i]), "sl": float(base.stop[i]),
                "tp": float(base.entry[i] + base.direction[i] * risk_price[i]), "exit": float(base.exit[i]),
                "reason": exit_reason(int(base.outcome[i]), int(base.exit_time[i]), label.hold),
                "r_gross": float(base.r_gross[i]), "cost_r": float((cost_price(s) + swap * nights) / risk_price[i]),
                "nights": nights, "r_net": float(base.r_net[i]), "forecast_edge": float(pred_by_time.get(int(base.signal_time[i]), np.nan)),
                "vol_ratio": float(f_after[np.searchsorted(r.time, base.signal_time[i])]),
                "take": bool(take[i]), "risk_pct": float(risk[i]),
            })
    out["records"] = records
    return out


def portfolio(res: dict, with_overlay: bool) -> dict:
    out = {}
    for seg_name in SEGMENTS:
        recs = [r for r in res["records"] if r["enabled"] and r["segment"] == seg_name and (r["take"] or not with_overlay)]
        if not recs:
            out[seg_name] = {"trades": 0}
            continue
        r_net = np.array([r["r_net"] for r in recs])
        risk = np.array([r["risk_pct"] if with_overlay else 1.0 for r in recs])
        exit_t = np.array([r["exit_time"] for r in recs])
        out[seg_name] = {"trades": len(recs), "win_rate": round(float((r_net > 0).mean()), 4), "avg_r": round(float(r_net.mean()), 4),
                         **money(r_net, risk, exit_t)}
    return out


def digits(symbol: str) -> int:
    return 2 if symbol.startswith("XAU") else 3 if symbol.endswith("JPY") else 5


def write_txt(results: dict, path: Path) -> None:
    L = []
    w = L.append
    w("EDGE HUNTER - EDGE ML trade log (research only; not used by the app)")
    w("=" * 120)
    w("Data: M1 MetaQuotes-Demo, 6 symbols, 2020-01 -> 2026-09-28, converted to UTC.")
    w("Models: trained on 2020-01 -> 2023-11-14 only; not changed afterwards. Overlay rule chosen on 2021-2023 validation folds only.")
    w("Periods: OOS 2023-11-14 -> 2024-12-31 | 2025-01 -> 2025-08 | NEW FORWARD 2025-09-01 -> 2026-09-28 (never seen by any model or choice).")
    w("Trade: market entry at the next minute open after the 15-minute decision bar; SL and TP both 2 x ATR14(M15) from the fill -> planned R:R = 1:1;")
    w("       stop-first when SL and TP fall in the same minute. A: exit by 4 h or 20:45 UTC. B1: exit after 12 h or Friday 20:45 UTC.")
    w("Costs: spread 0.1 pip (FX) / 0.10 $ (XAU) per trade; B1 swap estimate 0.5 pip (FX) / 0.50 $ (XAU) per night, Wednesday x3.")
    w("Money: 10,000 $ account, risk % of 10,000 $ per trade (no compounding). P&L $ = net R x risk $.")
    w("Overlay (volatility forecast): none = 1 % risk; size = 1 % x clip(median/forecast, 0.5, 1.5); filterNN = skip when the")
    w("       forecast is above the past 60 days' NN-th percentile. Skipped trades do not free the position slot (conservative).")
    w("R columns: R_gross = price move / SL distance; cost_R = (spread + swap x nights) / SL distance; R_net = R_gross - cost_R.")
    w("Symbols marked 'disabled' were switched off by validation in 2023; their trades are shown for information only and are")
    w("excluded from the portfolio lines.")
    w("")
    for name, res in results.items():
        w("#" * 120)
        w(f"VARIANT {VARIANTS[name]['title']}   (label {res['label']}, model size {res['hyper']})")
        w("#" * 120)
        w("Volatility forecast quality (R^2 of log realised volatility; higher is better):")
        for q in res["vol_quality"]:
            w(f"  {q['fold']:<42} model {q['r2_model']:+.3f} | hour-of-day baseline {q['r2_hour_of_day']:+.3f} | constant {q['r2_constant']:+.3f} | rows {q['rows']}")
        w("")
        head = f"  {'symbol':<8}{'status':<10}{'overlay':<15}{'period':<38}{'set':<9}{'trades':>7}{'win%':>7}{'avgR':>8}{'medR':>8}{'PF':>6}{'netR':>8}{'net $':>10}{'ret%':>7}{'maxDD%':>8}{'ret/DD':>7}{'skip':>5}{'avgRisk%':>9}"
        for s, sym in res["symbols"].items():
            re_ = sym["reselection_today"]
            w(f"  {s}: saved model = {sym['architecture']} keep {sym['keep']} ({'enabled' if sym['enabled_by_validation'] else 'disabled'}); "
              f"re-running the validation choice today would give {re_['architecture']} keep {re_['keep']} ({'enabled' if re_['enabled'] else 'disabled'})")
            w(head)
            for seg_name in SEGMENTS:
                for kind in ("without_overlay", "with_overlay"):
                    m = sym["segments"][seg_name][kind]
                    if not m.get("trades"):
                        w(f"  {s:<8}{'':<10}{'':<15}{SEGMENT_TITLES[seg_name]:<38}{kind[:7]:<9}{0:>7}")
                        continue
                    status = "enabled" if sym["enabled_by_validation"] else "disabled"
                    w(f"  {s:<8}{status:<10}{sym['overlay']:<15}{SEGMENT_TITLES[seg_name]:<38}{('plain' if kind == 'without_overlay' else 'overlay'):<9}"
                      f"{m['trades']:>7}{m['win_rate'] * 100:>7.1f}{m['avg_r']:>+8.3f}{m['median_r']:>+8.3f}{(m['profit_factor'] or 0):>6.2f}{m['net_r']:>+8.1f}"
                      f"{m['net_profit_usd']:>+10.0f}{m['net_return_pct']:>+7.1f}{m['max_dd_pct']:>8.1f}{(m['return_over_dd'] or 0):>7.2f}{m['skipped_by_overlay']:>5}{m['avg_risk_pct']:>9.2f}")
            w("")
        for kind, flag in (("plain (1 % risk, no overlay)", False), ("with the validation-chosen overlay", True)):
            w(f"  PORTFOLIO of validation-enabled symbols, {kind}:")
            for seg_name, m in portfolio(res, flag).items():
                if m.get("trades"):
                    w(f"    {SEGMENT_TITLES[seg_name]:<38} trades {m['trades']:>5}  win {m['win_rate'] * 100:5.1f}%  avg R {m['avg_r']:+.3f}  net {m['net_profit_usd']:+9.0f} $ ({m['net_return_pct']:+.1f} %)  max DD {m['max_dd_pct']:.1f} %  ret/DD {m['return_over_dd']}")
            w("")
    w("")
    w("=" * 120)
    w("TRADE LIST")
    w("=" * 120)
    cols = (f"{'#':>5} {'signal (UTC)':<16} {'entry (UTC)':<16} {'dir':<4} {'entry':>11} {'SL':>11} {'TP':>11} {'R:R':>4} {'exit (UTC)':<16} {'exit':>11} "
            f"{'reason':<7} {'min':>5} {'R_gross':>8} {'cost_R':>7} {'nights':>6} {'R_net':>7} {'vol_f':>6} {'overlay':<9} {'risk%':>6} {'P&L $':>9} {'equity $':>10}")
    for name, res in results.items():
        by_symbol: dict = {}
        for rec in res["records"]:
            by_symbol.setdefault(rec["symbol"], []).append(rec)
        for s, recs in by_symbol.items():
            sym = res["symbols"][s]
            for seg_name in SEGMENTS:
                seg_recs = sorted((r for r in recs if r["segment"] == seg_name), key=lambda r: r["entry_time"])
                w("")
                w(f"--- {name} | {s} | {'enabled' if sym['enabled_by_validation'] else 'DISABLED (shown for information)'} | overlay {sym['overlay']} | {SEGMENT_TITLES[seg_name]} | {len(seg_recs)} trades")
                w(cols)
                equity = CAPITAL
                dg = digits(s)
                for k, r in enumerate(seg_recs, 1):
                    if r["take"]:
                        pnl = r["r_net"] * r["risk_pct"] / 100 * CAPITAL
                        equity += pnl
                        act, risk_txt, pnl_txt = ("taken" if sym["overlay"] == "none" or abs(r["risk_pct"] - 1) < 1e-9 else f"x{r['risk_pct']:.2f}"), f"{r['risk_pct']:.2f}", f"{pnl:+.2f}"
                    else:
                        act, risk_txt, pnl_txt = "SKIPPED", "0.00", "0.00"
                    w(f"{k:>5} {iso(r['signal']):<16} {iso(r['entry_time']):<16} {r['dir']:<4} {r['entry']:>11.{dg}f} {r['sl']:>11.{dg}f} {r['tp']:>11.{dg}f} {'1:1':>4} "
                      f"{iso(r['exit_time']):<16} {r['exit']:>11.{dg}f} {r['reason']:<7} {(r['exit_time'] - r['entry_time']) // 60:>5} {r['r_gross']:>+8.3f} {r['cost_r']:>7.3f} "
                      f"{r['nights']:>6} {r['r_net']:>+7.3f} {r['vol_ratio']:>6.2f} {act:<9} {risk_txt:>6} {pnl_txt:>9} {equity:>10.2f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    started = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - started:6.0f}s] {msg}", flush=True)  # noqa: E731
    symbols = {s: load_symbol(RAW / s, CACHE) for s in discover_symbols(RAW) if s in TRAINING_SYMBOLS}
    results = {name: run_variant(name, cfg, symbols, log) for name, cfg in VARIANTS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    slim = {name: {k: v for k, v in res.items() if k != "records"} | {"portfolio_plain": portfolio(res, False), "portfolio_overlay": portfolio(res, True)}
            for name, res in results.items()}
    (RESULTS / "ML_EDGE_V3_VOL.json").write_text(json.dumps(slim, indent=1, default=str), encoding="utf-8")
    write_txt(results, RESULTS / "ML_EDGE_TRADES.txt")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

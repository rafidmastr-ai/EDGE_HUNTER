"""Offline check of the live EDGE ML path before it is used (docs/results/EDGE_ML_LIVE_CHECK.md).

1. Activity features: the live feed has no tick volume / spread, so live decisions use
   tick_volume_z = spread_z = 0. Compare those decisions with the research decisions on
   the new forward period (2025-09-01 -> 2026-09-28): agreement and trade results.
2. Threshold window: the live service decides over the last 80 days only; compare with the
   continuous research run at sampled decision bars.
3. Store replay: at sampled bars, rebuild the features from only the last 150 days of M1
   (what the live store keeps) and compare features and decision with the full history.

    python scripts/edge_ml_live_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_edge.data import SymbolData, load_symbol  # noqa: E402
from app.ml_edge.features import build_all  # noqa: E402
from app.ml_edge.live import ACTIVITY_FEATURES, SYMBOLS, VARIANTS  # noqa: E402
from app.ml_edge.live_store import KEEP_DAYS  # noqa: E402
from app.ml_edge.model import SymbolModel  # noqa: E402
from app.ml_edge.walkforward import DEV_END, execute, ts  # noqa: E402
from app.research.intraday import DAY, Bars, full_metrics, make_orders  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
CACHE = PROJECT_ROOT / "data" / "processed" / "edge_ml_cache"
MODELS = PROJECT_ROOT / "models" / "edge_ml"
OUT = PROJECT_ROOT / "docs" / "results" / "EDGE_ML_LIVE_CHECK.json"
NEW_FORWARD = ts("2025-09-01")
HISTORY_DAYS = 80
SAMPLES_WINDOW = 60
SAMPLES_REPLAY = 6


def trades_for(symbol_data, model, times, direction, atr):
    sel = direction != 0
    n = int(sel.sum())
    orders = make_orders(times[sel], direction[sel], np.full(n, np.nan), np.full(n, np.nan),
                         stop_distance=model.label.barrier_atr * atr[sel], target_rr=np.ones(n))
    return execute(symbol_data, orders, model.label)


def truncate(data: SymbolData, start: int, end: int) -> SymbolData:
    m = data.m1
    k = (m.open_time >= start) & (m.open_time < end)
    return SymbolData(data.symbol, Bars(m.open_time[k], m.open[k], m.high[k], m.low[k], m.close[k], m.period),
                      data.tick_volume[k], data.spread[k])


def zero_activity(frame) -> np.ndarray:
    x = frame.x.copy()
    for name in ACTIVITY_FEATURES:
        x[:, frame.names.index(name)] = 0.0
    return x


def main() -> int:
    rng = np.random.default_rng(7)
    symbols = {s: load_symbol(RAW / s, CACHE) for s in SYMBOLS}
    report: dict = {"new_forward_start": "2025-09-01", "models": {}}
    for variant, cfg in VARIANTS.items():
        frames = build_all(symbols, session_start_hour=cfg["session_start_hour"])
        for folder in sorted((MODELS / variant).iterdir()):
            model = SymbolModel.load(folder)
            s, key = model.symbol, f"{variant}/{model.symbol}"
            frame = frames[s]
            after = frame.tradable & (frame.close_time > DEV_END)
            times, atr = frame.close_time[after], frame.atr[after]
            d_research, _ = model.decide(times, frame.x[after], atr)
            d_live, _ = model.decide(times, zero_activity(frame)[after], atr)
            fwd = times >= NEW_FORWARD
            either = fwd & ((d_research != 0) | (d_live != 0))
            same = either & (d_research == d_live)
            t_res = trades_for(symbols[s], model, times[fwd], d_research[fwd], atr[fwd])
            t_live = trades_for(symbols[s], model, times[fwd], d_live[fwd], atr[fwd])
            item = {
                "signals_research": int((d_research[fwd] != 0).sum()),
                "signals_live_mode": int((d_live[fwd] != 0).sum()),
                "signal_agreement": round(float(same.sum() / max(1, either.sum())), 4),
                "research": {k: full_metrics(t_res).get(k) for k in ("trades", "win_rate", "avg_r", "profit_factor")},
                "live_mode": {k: full_metrics(t_live).get(k) for k in ("trades", "win_rate", "avg_r", "profit_factor")},
            }

            # 80-day decision window (the live service) vs the continuous research run
            idx = rng.choice(np.flatnonzero(fwd), size=min(SAMPLES_WINDOW, int(fwd.sum())), replace=False)
            match = 0
            for i in idx:
                w = (times > times[i] - HISTORY_DAYS * DAY) & (times <= times[i])
                d_w, _ = model.decide(times[w], frame.x[after][w], atr[w])
                match += int(d_w[-1] == d_research[i])
            item["window_80d_same_decision"] = f"{match}/{len(idx)}"
            item["window_80d_trade_samples"] = int((d_research[idx] != 0).sum())
            report["models"][key] = item
            print(key, json.dumps(item))

        # store replay: features from the last 150 days only
        replay = []
        grid = frames[SYMBOLS[0]]
        pool = np.flatnonzero(grid.tradable & (grid.close_time >= NEW_FORWARD))
        for i in rng.choice(pool, size=SAMPLES_REPLAY, replace=False):
            t = int(grid.close_time[i])
            short = build_all({s: truncate(d, t - KEEP_DAYS * DAY, t) for s, d in symbols.items()},
                              session_start_hour=cfg["session_start_hour"])
            worst = 0.0
            for s in SYMBOLS:
                a = frames[s].x[np.searchsorted(frames[s].close_time, t)]
                j = np.searchsorted(short[s].close_time, t)
                if j >= len(short[s].close_time) or short[s].close_time[j] != t:
                    continue
                b = short[s].x[j]
                ok = np.isfinite(a) & np.isfinite(b)
                worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0)
                if np.isfinite(a).sum() != np.isfinite(b).sum():
                    worst = float("inf")
            replay.append({"bar": t, "max_abs_feature_diff": worst})
        report[f"replay_{variant}"] = replay
        print(variant, "replay", replay)

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

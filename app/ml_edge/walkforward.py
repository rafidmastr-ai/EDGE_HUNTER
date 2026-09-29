"""Walk-forward study: fold fitting with embargo, selection on validation only, final fit, trades.

Selection happens in two stages so that few choices are made per symbol:
1. label design and hyper-parameters, pooled over all symbols, by the global model's
   rank correlation (IC) between forecast and outcome on the validation folds;
2. per symbol: architecture and entry threshold, by net R on the validation folds.
OOS and the forward period are never used for any choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.ml_edge.data import SymbolData, cost_price
from app.ml_edge.features import FeatureFrame
from app.ml_edge.labels import LabelConfig, triple_barrier
from app.ml_edge.model import ARCHITECTURES, SymbolModel, edges, fit_global, make_regressor, out_of_fold_global, rolling_thresholds, stack
from app.research.intraday import DAY, ExecutionConfig, Trades, make_orders, simulate


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


DEV_END = ts("2023-11-14T23:02:00")
OOS_END = ts("2025-01-01")
FOLDS = ((ts("2021-01-01"), ts("2022-01-01")), (ts("2022-01-01"), ts("2023-01-01")), (ts("2023-01-01"), DEV_END))
EMBARGO = DAY
LABELS = (LabelConfig(1.0, 8), LabelConfig(1.0, 16), LabelConfig(2.0, 8), LabelConfig(2.0, 16))
KEEP_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
MIN_VALIDATION_TRADES = 150


@dataclass
class Rows:
    symbol: str
    time: np.ndarray
    x: np.ndarray
    atr: np.ndarray
    y: dict[str, np.ndarray] = field(default_factory=dict)

    def mask(self, start: int | None, end: int | None) -> np.ndarray:
        m = np.ones(len(self.time), bool)
        if start is not None:
            m &= self.time >= start
        if end is not None:
            m &= self.time < end
        return m


def build_rows(symbols: dict[str, SymbolData], frames: dict[str, FeatureFrame], labels=LABELS) -> dict[str, Rows]:
    rows = {}
    for name, frame in frames.items():
        f = frame.select(frame.tradable)
        r = Rows(name, f.close_time, f.x, f.atr)
        for label in labels:
            r.y[label.key] = triple_barrier(symbols[name].m1, f.close_time, f.atr, label)
        rows[name] = r
    return rows


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _train_mask(r: Rows, key: str, end: int) -> np.ndarray:
    return r.mask(None, end - EMBARGO) & np.isfinite(r.y[key])


def fit_models(rows: dict[str, Rows], key: str, hyper: str, train_end: int, archs=ARCHITECTURES, label_shift: dict | None = None) -> dict:
    """Fit the global model and the per-symbol models on rows before ``train_end`` (minus embargo).

    ``label_shift`` (null test) circularly shifts each symbol's training labels.
    """
    def labels_for(r: Rows, m: np.ndarray) -> np.ndarray:
        y = r.y[key][m]
        if label_shift:
            y = np.roll(y, int(len(y) * label_shift[r.symbol]))
        return y

    masks = {s: _train_mask(r, key, train_end) for s, r in rows.items()}
    gx = np.vstack([rows[s].x[masks[s]] for s in rows])
    gy = np.concatenate([labels_for(rows[s], masks[s]) for s in rows])
    gt = np.concatenate([rows[s].time[masks[s]] for s in rows])
    out: dict = {"global": fit_global(gx, gy, hyper), "symbol": {}, "stacked": {}}
    oof = out_of_fold_global(gx, gy, gt, hyper) if "stacked" in archs else None
    start = 0
    for s, r in rows.items():
        n = int(masks[s].sum())
        x, y = r.x[masks[s]], labels_for(r, masks[s])
        if "symbol" in archs:
            out["symbol"][s] = make_regressor(hyper, symbol_level=True).fit(x, y)
        if "stacked" in archs:
            out["stacked"][s] = make_regressor(hyper, symbol_level=True).fit(stack(x, oof[start : start + n]), y)
        start += n
    return out


def forecast(models: dict, arch: str, symbol: str, x: np.ndarray) -> np.ndarray:
    if arch == "global":
        return models["global"].predict(x)
    if arch == "symbol":
        return models["symbol"][symbol].predict(x)
    return models["stacked"][symbol].predict(stack(x, models["global"].predict(x)))


def stage1(rows: dict[str, Rows], hypers, labels=LABELS, progress=print) -> dict:
    """IC of the global model on validation folds for every (label, hyper)."""
    table = {}
    for label in labels:
        for hyper in hypers:
            ics = []
            for train_end, val_end in FOLDS:
                models = fit_models(rows, label.key, hyper, train_end, archs=("global",))
                for s, r in rows.items():
                    m = r.mask(train_end, val_end)
                    ics.append(spearman(forecast(models, "global", s, r.x[m]), r.y[label.key][m]))
            table[(label.key, hyper)] = {"mean_ic": float(np.nanmean(ics)), "ics": [round(v, 4) for v in ics]}
            progress(f"  stage1 {label.key} {hyper}: mean IC {table[(label.key, hyper)]['mean_ic']:+.4f}")
    return table


def validation_forecasts(rows: dict[str, Rows], label: LabelConfig, hyper: str, label_shift: dict | None = None) -> dict:
    """{symbol: {arch: (fold_id, forecast)}} on the validation folds."""
    out: dict = {s: {a: ([], []) for a in ARCHITECTURES} for s in rows}
    for fold, (train_end, val_end) in enumerate(FOLDS):
        models = fit_models(rows, label.key, hyper, train_end, label_shift=label_shift)
        for s, r in rows.items():
            m = r.mask(train_end, val_end)
            for arch in ARCHITECTURES:
                out[s][arch][0].append(np.full(int(m.sum()), fold))
                out[s][arch][1].append(forecast(models, arch, s, r.x[m]))
    return {s: {a: (np.concatenate(v[0]), np.concatenate(v[1])) for a, v in d.items()} for s, d in out.items()}


def validation_mask(r: Rows) -> np.ndarray:
    return np.any([r.mask(train_end, val_end) for train_end, val_end in FOLDS], axis=0)


def choose_symbol(r: Rows, label: LabelConfig, fc: dict, cost: float) -> dict:
    """Pick architecture and keep-fraction for one symbol on its validation forecasts.

    Each fold's forecasts form their own series for the rolling threshold (the same rule
    the final model uses on OOS and live), so the first ``warmup_days`` of a fold cannot trade.
    """
    vm = validation_mask(r)
    times = r.time[vm]
    y = r.y[label.key][vm]
    cost_r = cost / (label.barrier_atr * r.atr[vm])
    candidates = []
    for arch in ARCHITECTURES:
        folds, f = fc[arch]
        edge, direction = edges(f, cost_r)
        net = direction * y - cost_r
        for keep in KEEP_FRACTIONS:
            threshold = np.full(len(edge), np.inf)
            for k in range(len(FOLDS)):
                fm = folds == k
                threshold[fm] = rolling_thresholds(times[fm], edge[fm], keep)
            sel = np.isfinite(net) & (edge > np.maximum(threshold, 0.0))
            if sel.sum() < MIN_VALIDATION_TRADES:
                continue
            per_fold = [float(net[sel & (folds == k)].mean()) if (sel & (folds == k)).any() else float("nan") for k in range(len(FOLDS))]
            candidates.append({"architecture": arch, "keep": keep, "trades": int(sel.sum()), "mean_net_r": float(net[sel].mean()),
                               "per_fold": per_fold, "_direction": np.where(sel, direction, 0)})
    if not candidates:
        return {"chosen": {"architecture": "global", "keep": KEEP_FRACTIONS[0], "trades": 0, "mean_net_r": float("nan"),
                           "per_fold": [], "enabled": False, "positive_folds": 0, "_direction": np.zeros(int(vm.sum()), int)},
                "candidates": [], "trials": 0}
    best = max(candidates, key=lambda c: c["mean_net_r"])
    positive_folds = sum(1 for v in best["per_fold"] if v > 0)
    best["enabled"] = bool(best["mean_net_r"] > 0 and positive_folds >= 2)
    best["positive_folds"] = positive_folds
    return {"chosen": best, "candidates": [{k: v for k, v in c.items() if k != "_direction"} for c in candidates], "trials": len(candidates)}


def final_models(rows: dict[str, Rows], label: LabelConfig, hyper: str, choices: dict, label_shift: dict | None = None) -> dict[str, SymbolModel]:
    models = fit_models(rows, label.key, hyper, DEV_END, label_shift=label_shift)
    out = {}
    for s, r in rows.items():
        c = choices[s]["chosen"]
        out[s] = SymbolModel(
            symbol=s,
            architecture=c["architecture"],
            hyper=hyper,
            label=label,
            feature_names=(),
            keep=c["keep"],
            enabled=c["enabled"],
            cost_price=cost_price(s, "base"),
            global_model=models["global"],
            symbol_model=models["stacked"][s] if c["architecture"] == "stacked" else models["symbol"].get(s),
        )
    return out


def orders_from(r: Rows, direction: np.ndarray, label: LabelConfig, mask: np.ndarray):
    sel = mask & (direction != 0)
    n = int(sel.sum())
    return make_orders(
        r.time[sel], direction[sel], np.full(n, np.nan), np.full(n, np.nan),
        stop_distance=label.barrier_atr * r.atr[sel], target_rr=np.ones(n),
    )


def execute(symbol_data: SymbolData, orders, label: LabelConfig, level: str = "base", one_position: bool = True) -> Trades:
    config = ExecutionConfig(cost=cost_price(symbol_data.symbol, level), horizon_bars=label.horizon_bars * 15, one_position=one_position)
    return simulate(symbol_data.m1, orders, config)


__all__ = [
    "DEV_END", "EMBARGO", "FOLDS", "LABELS", "OOS_END", "Rows", "build_rows", "choose_symbol", "execute", "final_models",
    "fit_models", "forecast", "orders_from", "spearman", "stage1", "validation_forecasts", "validation_mask",
]

"""Volatility forecast over a trade's horizon, and risk overlays built on it.

Target: realised volatility of M1 log returns from the entry bar to the trade's time
limit (same horizon rules as the labels), times the entry price, divided by ATR14(M15)
at decision time -> a scale-free ratio that is comparable across symbols. The model
predicts log(ratio) from the decision-time features only.

Overlays use the forecast f relative to the forecasts of the preceding 60 days
(``rolling_thresholds``), so they never need a fixed calibration:
- ``none``: 1 % risk;
- ``size``: risk = 1 % x clip(median(f, past 60d) / f, 0.5, 1.5);
- ``filter90`` / ``filter80``: skip when f is above the past 60 days' 90th / 80th percentile;
- ``size+filter90``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from app.ml_edge.labels import SESSION_EXIT, LabelConfig
from app.ml_edge.model import rolling_thresholds
from app.research.intraday import DAY, Bars, friday_cutoff

OVERLAYS = ("none", "size", "filter90", "filter80", "size+filter90")
BASE_RISK_PCT = 1.0


def horizon_bounds(m1: Bars, close_time: np.ndarray, label: LabelConfig, decision_minutes: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) M1 indices of the trade window for each decision (end exclusive; -1 when no entry)."""
    t = m1.open_time
    starts = np.searchsorted(t, close_time, "left")
    ends = np.full(len(close_time), -1, dtype=np.int64)
    for i, start in enumerate(starts.tolist()):
        if start >= len(t):
            continue
        if label.hold:
            if t[start] - close_time[i] > 3600:
                continue
            deadline = min(int(t[start]) + label.max_hold_minutes * 60, friday_cutoff(int(t[start]), (20, 45)))
            end = int(np.searchsorted(t, deadline, "left"))
        else:
            if t[start] // DAY != close_time[i] // DAY:
                continue
            end = min(int(np.searchsorted(t, int(close_time[i] // DAY * DAY + SESSION_EXIT), "left")), start + label.horizon_bars * decision_minutes)
        if end - start >= 2:
            ends[i] = end
    return starts.astype(np.int64), ends


def realised_vol_ratio(m1: Bars, close_time: np.ndarray, atr: np.ndarray, label: LabelConfig) -> np.ndarray:
    """Realised volatility over the trade window in ATR units (NaN when no window)."""
    log_close = np.log(m1.close)
    step = np.diff(log_close, prepend=log_close[0])
    csum = np.concatenate([[0.0], np.cumsum(step**2)])
    starts, ends = horizon_bounds(m1, close_time, label)
    out = np.full(len(close_time), np.nan)
    ok = (ends > starts + 1) & np.isfinite(atr) & (atr > 0)
    s, e = starts[ok] + 1, ends[ok]  # returns inside the window (from the entry bar onwards)
    rv = np.sqrt(np.maximum(csum[e] - csum[s], 0.0))
    out[ok] = rv * m1.open[starts[ok]] / atr[ok]
    return out


def make_vol_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=500,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)


def forecast_quality(y_log: np.ndarray, pred_log: np.ndarray, hour: np.ndarray, train_y_log: np.ndarray, train_hour: np.ndarray) -> dict:
    """R^2 of the model vs a constant (training mean) and an hour-of-day mean baseline."""
    ok = np.isfinite(y_log) & np.isfinite(pred_log)
    y, p, h = y_log[ok], pred_log[ok], hour[ok]
    sst = float(((y - y.mean()) ** 2).sum())
    const = np.full(len(y), np.nanmean(train_y_log))
    means = {k: float(np.nanmean(train_y_log[train_hour == k])) for k in np.unique(train_hour)}
    by_hour = np.array([means.get(k, float(np.nanmean(train_y_log))) for k in h])

    def r2(pred: np.ndarray) -> float:
        return round(1.0 - float(((y - pred) ** 2).sum()) / sst, 4)

    return {"rows": int(ok.sum()), "r2_model": r2(p), "r2_constant": r2(const), "r2_hour_of_day": r2(by_hour),
            "corr_model": round(float(np.corrcoef(y, p)[0, 1]), 4)}


def overlay_actions(times: np.ndarray, forecast: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """(take: bool, risk_pct) per decision row for ``rule``; thresholds from the previous 60 days only."""
    if rule not in OVERLAYS:
        raise ValueError(f"unknown overlay {rule!r}")
    n = len(times)
    take = np.ones(n, bool)
    risk = np.full(n, BASE_RISK_PCT)
    if rule == "none":
        return take, risk
    if "filter" in rule:
        keep = 0.10 if "90" in rule else 0.20
        limit = rolling_thresholds(times, forecast, keep, warmup_days=20)
        take = ~(np.isfinite(limit) & (forecast > limit))
    if rule.startswith("size"):
        median = rolling_thresholds(times, forecast, 0.5, warmup_days=20)
        scale = np.where(np.isfinite(median) & (forecast > 0), median / forecast, 1.0)
        risk = BASE_RISK_PCT * np.clip(scale, 0.5, 1.5)
    return take, risk


@dataclass
class VolModel:
    label: LabelConfig
    feature_names: tuple[str, ...]
    regressor: HistGradientBoostingRegressor

    def forecast_ratio(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.regressor.predict(np.asarray(x, dtype=np.float32)))

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        blob = directory / "vol_model.joblib"
        joblib.dump(self.regressor, blob)
        manifest = {"label": self.label.__dict__, "feature_names": list(self.feature_names),
                    "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}
        (directory / "vol_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VolModel":
        import joblib

        manifest = json.loads((directory / "vol_manifest.json").read_text(encoding="utf-8"))
        blob = directory / "vol_model.joblib"
        if hashlib.sha256(blob.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError(f"volatility model checksum mismatch in {directory}")
        return cls(LabelConfig(**manifest["label"]), tuple(manifest["feature_names"]), joblib.load(blob))


__all__ = ["OVERLAYS", "VolModel", "forecast_quality", "horizon_bounds", "make_vol_regressor", "overlay_actions", "realised_vol_ratio"]

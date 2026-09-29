"""Global (all symbols) and per-symbol gradient-boosted models, decisions and persistence.

Architectures:
- ``global``: the model trained on all symbols' rows predicts directly.
- ``symbol``: a model trained only on this symbol's rows.
- ``stacked``: a symbol model whose inputs are the features plus the global forecast
  (trained on out-of-fold global forecasts to avoid leaking the global model's fit).

A saved :class:`SymbolModel` contains only this symbol's model plus a frozen copy of the
global model (shared knowledge); it never calls another symbol's model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from app.ml_edge.labels import LabelConfig

ARCHITECTURES = ("global", "symbol", "stacked")
HYPER = {
    "small": {"max_depth": 3, "max_iter": 200, "learning_rate": 0.05, "min_samples_leaf": 500, "l2_regularization": 1.0},
    "medium": {"max_depth": 5, "max_iter": 300, "learning_rate": 0.05, "min_samples_leaf": 300, "l2_regularization": 1.0},
    "large": {"max_depth": 7, "max_iter": 400, "learning_rate": 0.03, "min_samples_leaf": 200, "l2_regularization": 1.0},
}
SYMBOL_MIN_LEAF = 400  # stronger regularisation for single-symbol models (less data)


def make_regressor(hyper: str, *, symbol_level: bool = False) -> HistGradientBoostingRegressor:
    params = dict(HYPER[hyper])
    if symbol_level:
        params["min_samples_leaf"] = max(params["min_samples_leaf"], SYMBOL_MIN_LEAF)
    return HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=0, **params)


def fit_global(x: np.ndarray, y: np.ndarray, hyper: str) -> HistGradientBoostingRegressor:
    return make_regressor(hyper).fit(x, y)


def out_of_fold_global(x: np.ndarray, y: np.ndarray, time_: np.ndarray, hyper: str, blocks: int = 2) -> np.ndarray:
    """Global forecasts for training rows from models that did not see those rows (contiguous time blocks)."""
    order = np.argsort(time_, kind="stable")
    edges = np.linspace(0, len(order), blocks + 1).astype(int)
    out = np.empty(len(y))
    for k in range(blocks):
        held = order[edges[k] : edges[k + 1]]
        mask = np.ones(len(y), bool)
        mask[held] = False
        out[held] = make_regressor(hyper).fit(x[mask], y[mask]).predict(x[held])
    return out


def stack(x: np.ndarray, global_forecast: np.ndarray) -> np.ndarray:
    return np.column_stack([x, global_forecast]).astype(np.float32)


def edges(forecast: np.ndarray, cost_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(best net edge in R, direction +1/-1) for symmetric long/short barrier trades."""
    long_edge = forecast - cost_r
    short_edge = -forecast - cost_r
    direction = np.where(long_edge >= short_edge, 1, -1)
    return np.maximum(long_edge, short_edge), direction


@dataclass
class SymbolModel:
    symbol: str
    architecture: str
    hyper: str
    label: LabelConfig
    feature_names: tuple[str, ...]
    theta: float
    enabled: bool
    cost_price: float
    global_model: Any = None
    symbol_model: Any = None
    info: dict = field(default_factory=dict)

    def forecast(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.architecture == "global":
            return self.global_model.predict(x)
        if self.architecture == "symbol":
            return self.symbol_model.predict(x)
        return self.symbol_model.predict(stack(x, self.global_model.predict(x)))

    def decide(self, x: np.ndarray, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(direction +1/-1/0, forecast) for rows of features at decision time."""
        forecast = self.forecast(x)
        cost_r = self.cost_price / (self.label.barrier_atr * np.asarray(atr, dtype=float))
        edge, direction = edges(forecast, cost_r)
        trade = self.enabled & (edge > self.theta)
        return np.where(trade, direction, 0), forecast

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> Path:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        blob = directory / "model.joblib"
        joblib.dump({"global_model": self.global_model, "symbol_model": self.symbol_model}, blob)
        manifest = {
            "symbol": self.symbol,
            "architecture": self.architecture,
            "hyper": self.hyper,
            "label": asdict(self.label),
            "feature_names": list(self.feature_names),
            "theta": self.theta,
            "enabled": self.enabled,
            "cost_price": self.cost_price,
            "info": self.info,
            "model_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "SymbolModel":
        import joblib

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        blob = directory / "model.joblib"
        if hashlib.sha256(blob.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError(f"model checksum mismatch in {directory}")
        models = joblib.load(blob)
        return cls(
            symbol=manifest["symbol"],
            architecture=manifest["architecture"],
            hyper=manifest["hyper"],
            label=LabelConfig(**manifest["label"]),
            feature_names=tuple(manifest["feature_names"]),
            theta=float(manifest["theta"]),
            enabled=bool(manifest["enabled"]),
            cost_price=float(manifest["cost_price"]),
            global_model=models["global_model"],
            symbol_model=models["symbol_model"],
            info=manifest.get("info", {}),
        )


__all__ = ["ARCHITECTURES", "HYPER", "SymbolModel", "edges", "fit_global", "make_regressor", "out_of_fold_global", "stack"]

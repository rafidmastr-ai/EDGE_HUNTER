"""Machine learning for the Classic, SMC and ICT strategies (meta-labeling).

Each rule-based strategy keeps generating its own setups. A per-strategy model
learns, from labelled past setups, the probability that a new setup reaches its
target before its stop-loss. At analysis time a setup whose probability is below
the strategy's learned threshold is withheld (NO_SIGNAL with an explanation);
the others pass through with their probability attached.

Learning sources:
1. Historical OHLC CSV files (``data/raw``): every setup each strategy would have
   produced is labelled with the Phase 05 TP/SL semantics (OutcomeLabelEngine).
2. Live setups: after an analysis response has been sent, each actionable
   strategy setup is stored as a LIVE_TRADE LearningRecord (PENDING_OUTCOME).
   The training script later resolves it against the bars that followed and
   includes the completed record in the next training run.

Safety:
- Training happens only in ``scripts/train_strategies.py``; the web request path
  only performs read-only inference with an already approved model.
- Chronological TRAIN / VALIDATION / OOS split. A model becomes ACTIVE only if
  its filter improves the average R multiple on the untouched OOS segment;
  otherwise the strategy keeps running unfiltered.
- Models are stored as JSON coefficients with a SHA-256 checksum (no pickle).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.backtest.config import BacktestConfig
from app.data.schema import CanonicalOHLC
from app.db.database import Database
from app.features.engine import FeatureEngine
from app.features.models import MarketAnalysisSeries
from app.learning.labels import LearningOutcome, OutcomeLabelEngine, OutcomeResolutionState
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)
from app.learning.repository import LearningRecordRepository
from app.research.resampling import resample_ohlc
from app.strategies.models import SignalDirection, SignalState, StrategyContext, StrategySignal
from app.strategies.registry import StrategyRegistry

logger = logging.getLogger("edge_hunter.learning.strategy_ml")

FEATURE_SET_VERSION = "strategy-ml-features-v1"
MODEL_FORMAT_VERSION = "strategy-ml-model-v1"
LEARNED_STRATEGIES = ("Classic", "SMC", "ICT")
ANALYSIS_TIMEFRAMES = ("M5", "M15", "H1")
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
DEFAULT_HORIZON_BARS = 48
# Strategies only look back a handful of candles; features are precomputed on the
# full series, so a bounded window gives identical signals at O(1) cost per bar.
CONTEXT_WINDOW_BARS = 300

FEATURE_NAMES: tuple[str, ...] = (
    "rsi_directional",
    "macd_hist_atr_directional",
    "macd_atr_directional",
    "return_1_directional",
    "return_5_directional",
    "return_20_directional",
    "close_vs_ema20_atr_directional",
    "close_vs_ema50_atr_directional",
    "close_vs_ema200_atr_directional",
    "ema_spread_20_50_directional",
    "ema_alignment_directional",
    "body_ratio",
    "close_location_directional",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "atr_pct",
    "realized_volatility",
    "risk_reward",
    "stop_distance_atr",
    "target_distance_atr",
    "timeframe_m5",
    "timeframe_m15",
    "timeframe_h1",
)

STATUS_ACTIVE = "ACTIVE"
STATUS_REJECTED = "REJECTED"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class StrategyLearningError(RuntimeError):
    """Raised for invalid learning inputs or artifacts."""


# ---------------------------------------------------------------------------
# Features (shared by training and live inference)
# ---------------------------------------------------------------------------
def _number(values: Mapping[str, Any], key: str) -> float | None:
    value = values.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def signal_features(signal: StrategySignal, snapshot_values: Mapping[str, Any], timeframe: str) -> dict[str, float] | None:
    """Decision-time features of one actionable setup, scale-free and direction-aware.

    Only information available at the decision candle close is used. Returns
    ``None`` when required indicators are unavailable (e.g. warm-up).
    """
    if signal.state != SignalState.SIGNAL or signal.entry is None or signal.stop_loss is None or signal.target is None:
        return None
    direction = 1.0 if signal.direction == SignalDirection.BUY else -1.0
    close = _number(snapshot_values, "price.close")
    atr = _number(snapshot_values, "volatility.atr")
    rsi = _number(snapshot_values, "momentum.rsi")
    macd = _number(snapshot_values, "momentum.macd")
    macd_hist = _number(snapshot_values, "momentum.macd_histogram")
    ema20 = _number(snapshot_values, "trend.ema_20")
    ema50 = _number(snapshot_values, "trend.ema_50")
    ema200 = _number(snapshot_values, "trend.ema_200")
    required = (close, atr, rsi, macd, macd_hist, ema20, ema50, ema200)
    if any(item is None for item in required) or not atr or atr <= 0 or not close or close <= 0:
        return None

    candle_range = _number(snapshot_values, "structure.range") or 0.0
    upper = _number(snapshot_values, "structure.upper_wick") or 0.0
    lower = _number(snapshot_values, "structure.lower_wick") or 0.0
    close_location = _number(snapshot_values, "structure.close_location")
    close_location = 0.5 if close_location is None else close_location
    alignment = str(snapshot_values.get("trend.ema_alignment") or "").upper()
    alignment_value = 1.0 if alignment == "BULLISH" else -1.0 if alignment == "BEARISH" else 0.0
    tf = timeframe.upper()

    def clip(value: float, bound: float = 10.0) -> float:
        return max(-bound, min(bound, value))

    features = {
        "rsi_directional": direction * (rsi - 50.0) / 50.0,
        "macd_hist_atr_directional": clip(direction * macd_hist / atr),
        "macd_atr_directional": clip(direction * macd / atr),
        "return_1_directional": clip(direction * (_number(snapshot_values, "returns.simple_1") or 0.0) * 100.0),
        "return_5_directional": clip(direction * (_number(snapshot_values, "returns.simple_5") or 0.0) * 100.0),
        "return_20_directional": clip(direction * (_number(snapshot_values, "returns.simple_20") or 0.0) * 100.0),
        "close_vs_ema20_atr_directional": clip(direction * (close - ema20) / atr),
        "close_vs_ema50_atr_directional": clip(direction * (close - ema50) / atr),
        "close_vs_ema200_atr_directional": clip(direction * (close - ema200) / atr, 50.0),
        "ema_spread_20_50_directional": clip(direction * (_number(snapshot_values, "trend.ema_spread_20_50_pct") or 0.0)),
        "ema_alignment_directional": direction * alignment_value,
        "body_ratio": _number(snapshot_values, "structure.body_ratio") or 0.0,
        "close_location_directional": close_location if direction > 0 else 1.0 - close_location,
        "upper_wick_ratio": upper / candle_range if candle_range > 0 else 0.0,
        "lower_wick_ratio": lower / candle_range if candle_range > 0 else 0.0,
        "atr_pct": clip(_number(snapshot_values, "volatility.atr_pct") or 0.0),
        "realized_volatility": clip((_number(snapshot_values, "volatility.realized_log_std") or 0.0) * 100.0),
        "risk_reward": float(signal.risk_reward or 0.0),
        "stop_distance_atr": clip(abs(signal.entry - signal.stop_loss) / atr, 50.0),
        "target_distance_atr": clip(abs(signal.target - signal.entry) / atr, 50.0),
        "timeframe_m5": 1.0 if tf == "M5" else 0.0,
        "timeframe_m15": 1.0 if tf == "M15" else 0.0,
        "timeframe_h1": 1.0 if tf == "H1" else 0.0,
    }
    if not all(math.isfinite(value) for value in features.values()):
        return None
    return features


# ---------------------------------------------------------------------------
# Samples and labels
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategySample:
    strategy: str
    symbol: str
    timeframe: str
    timestamp: datetime
    features: Mapping[str, float]
    won: int
    r_multiple: float
    source: str  # "HISTORICAL" or "LIVE"


def _r_multiple(direction: LearningDirection, entry: float, stop: float, exit_price: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    move = (exit_price - entry) if direction == LearningDirection.BUY else (entry - exit_price)
    return move / risk


def _record_for_signal(
    signal: StrategySignal,
    *,
    symbol: str,
    timeframe: str,
    features: Mapping[str, float],
    source_type: LearningSourceType,
    record_id: str,
    confidence: float = 0.0,
    provenance: Mapping[str, Any] | None = None,
) -> LearningRecord:
    assert signal.entry is not None and signal.stop_loss is not None and signal.target is not None
    return LearningRecord(
        record_id=record_id,
        source_type=source_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
        decision_timestamp=signal.timestamp,
        entry_timestamp=signal.timestamp,
        direction=LearningDirection(signal.direction.value),
        entry_price=float(signal.entry),
        target=float(signal.target),
        stop_loss=float(signal.stop_loss),
        risk_reward=float(signal.risk_reward or 0.0),
        strategy_name=signal.strategy_name,
        strategy_variant=signal.variant,
        strategy_version=signal.variant,
        feature_set_version=FEATURE_SET_VERSION,
        parameters_snapshot={},
        feature_snapshot=LearningFeatureSnapshot(dict(features)),
        evidence_snapshot={"evidence": [str(item) for item in signal.evidence]},
        confidence=max(0.0, min(100.0, float(confidence))),
        market_regime=None,
        provenance_metadata=dict(provenance or {}),
    )


class HistoricalSampleBuilder:
    """Replay every strategy over historical OHLC and label each setup."""

    def __init__(
        self,
        registry: StrategyRegistry | None = None,
        *,
        horizon_bars: int = DEFAULT_HORIZON_BARS,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self.registry = registry or StrategyRegistry.default()
        self.horizon_bars = max(2, int(horizon_bars))
        self.label_engine = OutcomeLabelEngine(backtest_config or BacktestConfig())
        self.feature_engine = FeatureEngine()

    def samples_from_bars(self, symbol: str, timeframe: str, bars: Sequence[CanonicalOHLC]) -> list[StrategySample]:
        records = tuple(bars)
        if len(records) < 50:
            return []
        analysis = self.feature_engine.compute(records, symbol, timeframe)
        snapshots = analysis.snapshots
        samples: list[StrategySample] = []
        last_decision = len(records) - 1 - self.horizon_bars
        for index in range(last_decision + 1):
            current = snapshots[index]
            if not current.warmup_complete:
                continue
            start = max(0, index + 1 - CONTEXT_WINDOW_BARS)
            window = MarketAnalysisSeries(
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                snapshots=snapshots[start : index + 1],
                definitions=analysis.definitions,
                warmup_bars_required=analysis.warmup_bars_required,
                volume_features_enabled=analysis.volume_features_enabled,
                engine_version=analysis.engine_version,
            )
            context = StrategyContext.from_series(records[start : index + 1], window)
            for strategy in self.registry.strategies:
                signal = strategy.generate(context)
                if signal.state != SignalState.SIGNAL or signal.strategy_name not in LEARNED_STRATEGIES:
                    continue
                sample = self._label(signal, symbol, timeframe, current.values, records[index + 1 : index + 1 + self.horizon_bars])
                if sample is not None:
                    samples.append(sample)
        return samples

    def _label(
        self,
        signal: StrategySignal,
        symbol: str,
        timeframe: str,
        values: Mapping[str, Any],
        future: Sequence[CanonicalOHLC],
    ) -> StrategySample | None:
        features = signal_features(signal, values, timeframe)
        if features is None:
            return None
        record = _record_for_signal(
            signal,
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            record_id=f"hist-{symbol}-{timeframe}-{signal.strategy_name}-{signal.timestamp.isoformat()}",
        )
        resolution = self.label_engine.evaluate(record, future, data_complete=True)
        if resolution.state != OutcomeResolutionState.COMPLETED or resolution.exit_price is None:
            return None
        return StrategySample(
            strategy=signal.strategy_name,
            symbol=symbol.upper(),
            timeframe=timeframe,
            timestamp=signal.timestamp,
            features=features,
            won=1 if resolution.outcome == LearningOutcome.TP_BEFORE_SL else 0,
            r_multiple=_r_multiple(record.direction, record.entry_price, record.stop_loss, resolution.exit_price),
            source="HISTORICAL",
        )


def base_interval_minutes(bars: Sequence[CanonicalOHLC]) -> int:
    """Median spacing of a bar series in minutes (the CSV's native timeframe)."""
    if len(bars) < 3:
        return 1
    deltas = [
        (bars[i].timestamp - bars[i - 1].timestamp).total_seconds() / 60.0
        for i in range(1, min(len(bars), 5001))
        if bars[i].timestamp > bars[i - 1].timestamp
    ]
    return max(1, int(round(statistics.median(deltas)))) if deltas else 1


def valid_ohlc(bars: Iterable[CanonicalOHLC]) -> list[CanonicalOHLC]:
    """Drop rows with impossible OHLC geometry (the feature engine rejects them)."""
    return [
        bar
        for bar in bars
        if bar.high >= max(bar.open, bar.close, bar.low) and bar.low <= min(bar.open, bar.close, bar.high) and bar.low > 0
    ]


def timeframe_bars(bars: Sequence[CanonicalOHLC], timeframe: str) -> tuple[CanonicalOHLC, ...] | None:
    """Bars at ``timeframe`` built from a CSV's native bars, or None if coarser data."""
    base = base_interval_minutes(bars)
    target = TIMEFRAME_MINUTES[timeframe]
    if target < base or target % base:
        return None
    if target == base:
        return tuple(bars)
    return resample_ohlc(bars, timeframe)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyModel:
    strategy: str
    status: str
    reason: str
    feature_names: tuple[str, ...] = FEATURE_NAMES
    mean: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    coef: tuple[float, ...] = ()
    intercept: float = 0.0
    threshold: float = 0.5
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status == STATUS_ACTIVE and len(self.coef) == len(self.feature_names)

    def probability(self, features: Mapping[str, float]) -> float:
        z = self.intercept
        for name, mean, scale, weight in zip(self.feature_names, self.mean, self.scale, self.coef):
            z += weight * ((float(features.get(name, 0.0)) - mean) / (scale or 1.0))
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "reason": self.reason,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coef": list(self.coef),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyModel":
        names = tuple(str(item) for item in payload.get("feature_names", ()))
        model = cls(
            strategy=str(payload["strategy"]),
            status=str(payload["status"]),
            reason=str(payload.get("reason", "")),
            feature_names=names,
            mean=tuple(float(item) for item in payload.get("mean", ())),
            scale=tuple(float(item) for item in payload.get("scale", ())),
            coef=tuple(float(item) for item in payload.get("coef", ())),
            intercept=float(payload.get("intercept", 0.0)),
            threshold=float(payload.get("threshold", 0.5)),
            metrics=dict(payload.get("metrics", {})),
        )
        if model.status == STATUS_ACTIVE:
            if names != FEATURE_NAMES:
                raise StrategyLearningError(f"{model.strategy}: model feature schema does not match {FEATURE_SET_VERSION}")
            if not (len(model.mean) == len(model.scale) == len(model.coef) == len(names)):
                raise StrategyLearningError(f"{model.strategy}: inconsistent model dimensions")
            if not 0.0 < model.threshold < 1.0:
                raise StrategyLearningError(f"{model.strategy}: threshold outside (0, 1)")
        return model


@dataclass(frozen=True)
class StrategyLearningBundle:
    trained_at: datetime
    models: Mapping[str, StrategyModel]
    data_summary: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "trained_at": self.trained_at.astimezone(timezone.utc).isoformat(),
            "data_summary": dict(self.data_summary),
            "models": {name: model.to_dict() for name, model in sorted(self.models.items())},
        }

    def save(self, path: Path) -> Path:
        payload = self.payload()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        document = {"sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "payload": payload}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)  # atomic: the web app never sees a half-written file
        return path

    @classmethod
    def load(cls, path: Path) -> "StrategyLearningBundle":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise StrategyLearningError("model file has no payload")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != document.get("sha256"):
            raise StrategyLearningError("model file checksum mismatch")
        if payload.get("format_version") != MODEL_FORMAT_VERSION or payload.get("feature_set_version") != FEATURE_SET_VERSION:
            raise StrategyLearningError("model file format/feature version is not supported")
        models = {name: StrategyModel.from_dict(item) for name, item in payload.get("models", {}).items()}
        return cls(
            trained_at=datetime.fromisoformat(payload["trained_at"]),
            models=models,
            data_summary=payload.get("data_summary", {}),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class StrategyLearningTrainer:
    """Train one logistic-regression filter per strategy with an OOS promotion gate."""

    THRESHOLD_GRID = tuple(round(0.30 + 0.02 * step, 2) for step in range(21))  # 0.30 .. 0.70

    def __init__(
        self,
        *,
        min_samples: int = 200,
        min_oos_kept: int = 20,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
        min_keep_fraction: float = 0.2,
        regularization_c: float = 0.5,
        min_improvement_r: float = 0.02,
    ) -> None:
        if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
            raise ValueError("invalid chronological split fractions")
        self.min_samples = min_samples
        self.min_oos_kept = min_oos_kept
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.min_keep_fraction = min_keep_fraction
        self.regularization_c = regularization_c
        # Minimum gain in average R per setup; smaller gains are treated as noise.
        self.min_improvement_r = min_improvement_r

    def train(self, samples: Iterable[StrategySample], *, data_summary: Mapping[str, Any] | None = None) -> StrategyLearningBundle:
        grouped: dict[str, list[StrategySample]] = {name: [] for name in LEARNED_STRATEGIES}
        for sample in samples:
            if sample.strategy in grouped:
                grouped[sample.strategy].append(sample)
        models = {name: self._train_one(name, items) for name, items in grouped.items()}
        return StrategyLearningBundle(trained_at=datetime.now(timezone.utc), models=models, data_summary=dict(data_summary or {}))

    def _train_one(self, strategy: str, samples: list[StrategySample]) -> StrategyModel:
        ordered = sorted(samples, key=lambda item: (item.timestamp, item.symbol, item.timeframe))
        counts = {
            "samples": len(ordered),
            "historical": sum(1 for item in ordered if item.source == "HISTORICAL"),
            "live": sum(1 for item in ordered if item.source == "LIVE"),
        }
        if len(ordered) < self.min_samples:
            return StrategyModel(strategy, STATUS_INSUFFICIENT_DATA, f"needs at least {self.min_samples} labelled setups", metrics=counts)

        n = len(ordered)
        train_end = int(n * self.train_fraction)
        validation_end = int(n * (self.train_fraction + self.validation_fraction))
        train, validation, oos = ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]
        if len({item.won for item in train}) < 2:
            return StrategyModel(strategy, STATUS_INSUFFICIENT_DATA, "training segment has a single outcome class", metrics=counts)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        def matrix(items: Sequence[StrategySample]) -> list[list[float]]:
            return [[float(item.features.get(name, 0.0)) for name in FEATURE_NAMES] for item in items]

        scaler = StandardScaler().fit(matrix(train))  # fitted on TRAIN only
        classifier = LogisticRegression(C=self.regularization_c, max_iter=2000)
        classifier.fit(scaler.transform(matrix(train)), [item.won for item in train])

        scale = tuple(float(value) if value > 0 else 1.0 for value in scaler.scale_)
        draft = StrategyModel(
            strategy=strategy,
            status=STATUS_REJECTED,
            reason="",
            mean=tuple(float(value) for value in scaler.mean_),
            scale=scale,
            coef=tuple(float(value) for value in classifier.coef_[0]),
            intercept=float(classifier.intercept_[0]),
        )

        def evaluate(items: Sequence[StrategySample], threshold: float) -> dict[str, float]:
            kept = [item for item in items if draft.probability(item.features) >= threshold]
            return {
                "count": len(items),
                "kept": len(kept),
                "baseline_avg_r": round(_mean([item.r_multiple for item in items]), 4),
                "baseline_win_rate": round(_mean([item.won for item in items]), 4),
                "filtered_avg_r": round(_mean([item.r_multiple for item in kept]), 4),
                "filtered_win_rate": round(_mean([item.won for item in kept]), 4),
            }

        # Threshold chosen on VALIDATION only; OOS stays untouched until the gate.
        minimum_kept = max(10, int(len(validation) * self.min_keep_fraction))
        best_threshold, best = None, None
        for threshold in self.THRESHOLD_GRID:
            result = evaluate(validation, threshold)
            if result["kept"] < minimum_kept:
                continue
            if best is None or result["filtered_avg_r"] > best["filtered_avg_r"]:
                best_threshold, best = threshold, result
        metrics: dict[str, Any] = {**counts, "train": len(train)}
        if best is None or best_threshold is None:
            return replace(draft, reason="no threshold keeps enough validation setups", metrics=metrics)

        validation_result = best
        oos_result = evaluate(oos, best_threshold)
        metrics.update({"threshold": best_threshold, "validation": validation_result, "oos": oos_result})
        margin = self.min_improvement_r
        improved_validation = validation_result["filtered_avg_r"] >= validation_result["baseline_avg_r"] + margin
        improved_oos = oos_result["filtered_avg_r"] >= oos_result["baseline_avg_r"] + margin
        metrics["min_improvement_r"] = margin
        if not improved_validation:
            return replace(draft, threshold=best_threshold, reason="filter does not improve validation expectancy", metrics=metrics)
        if oos_result["kept"] < self.min_oos_kept:
            return replace(draft, threshold=best_threshold, reason="too few out-of-sample setups kept by the filter", metrics=metrics)
        if not improved_oos:
            return replace(draft, threshold=best_threshold, reason="filter does not improve out-of-sample expectancy", metrics=metrics)
        return replace(draft, status=STATUS_ACTIVE, threshold=best_threshold, reason="out-of-sample expectancy improved", metrics=metrics)


# ---------------------------------------------------------------------------
# Live inference (read-only)
# ---------------------------------------------------------------------------
class StrategyLearningFilter:
    """Apply the approved per-strategy models to strategy outputs.

    Reloads the model file automatically when the training script replaces it,
    so a new training run takes effect without restarting the web server. Any
    missing, corrupt or tampered model file leaves the strategies unfiltered.
    """

    def __init__(self, model_path: Path, *, enabled: bool = True, reload_interval_seconds: float = 30.0) -> None:
        self.model_path = Path(model_path)
        self.enabled = enabled
        self.reload_interval = max(0.0, float(reload_interval_seconds))
        self._lock = threading.Lock()
        self._bundle: StrategyLearningBundle | None = None
        self._mtime: float | None = None
        self._checked_at = -math.inf
        self._load_error: str | None = None

    def _current(self) -> StrategyLearningBundle | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            if now - self._checked_at < self.reload_interval:
                return self._bundle
            self._checked_at = now
            try:
                mtime = self.model_path.stat().st_mtime
            except OSError:
                self._bundle, self._mtime, self._load_error = None, None, None
                return None
            if mtime != self._mtime:
                try:
                    self._bundle = StrategyLearningBundle.load(self.model_path)
                    self._load_error = None
                    logger.info("strategy ML models loaded", extra={"active": self.active_strategies(locked=True)})
                except (OSError, ValueError, KeyError, StrategyLearningError) as exc:
                    self._bundle = None
                    self._load_error = type(exc).__name__
                    logger.warning("strategy ML model file rejected; strategies run unfiltered", extra={"error": str(exc)[:200]})
                self._mtime = mtime
            return self._bundle

    def active_strategies(self, *, locked: bool = False) -> list[str]:
        bundle = self._bundle if locked else self._current()
        if bundle is None:
            return []
        return sorted(name for name, model in bundle.models.items() if model.active)

    def status(self) -> dict[str, Any]:
        bundle = self._current()
        return {
            "enabled": self.enabled,
            "model_loaded": bundle is not None,
            "trained_at": bundle.trained_at.isoformat() if bundle else None,
            "active_strategies": self.active_strategies(),
            "strategies": {
                name: {"status": model.status, "reason": model.reason, "threshold": model.threshold}
                for name, model in (bundle.models.items() if bundle else ())
            },
            "load_error": self._load_error,
        }

    def apply(self, signals: Sequence[StrategySignal], context: StrategyContext) -> tuple[StrategySignal, ...]:
        bundle = self._current()
        if bundle is None:
            return tuple(signals)
        output: list[StrategySignal] = []
        for signal in signals:
            model = bundle.models.get(signal.strategy_name)
            if signal.state != SignalState.SIGNAL or model is None or not model.active:
                output.append(signal)
                continue
            features = signal_features(signal, context.current.values, context.timeframe)
            if features is None:
                output.append(signal)
                continue
            probability = round(model.probability(features), 4)
            learning_meta = {
                "ml_probability": probability,
                "ml_threshold": model.threshold,
                "ml_model_trained_at": bundle.trained_at.isoformat(),
            }
            if probability >= model.threshold:
                output.append(replace(signal, metadata={**dict(signal.metadata), **learning_meta, "ml_decision": "accepted"}))
                continue
            output.append(
                StrategySignal(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    direction=SignalDirection.NO_SIGNAL,
                    state=SignalState.NO_SIGNAL,
                    entry=None,
                    stop_loss=None,
                    target=None,
                    risk_reward=None,
                    entry_logic="Setup withheld by the strategy's learned filter.",
                    invalidation="No active setup.",
                    stop_loss_logic="No stop-loss generated.",
                    target_logic="No target generated.",
                    evidence=(f"ml_filter_rejected:p={probability:.2f}<{model.threshold:.2f}",),
                    score_inputs={},
                    strategy_name=signal.strategy_name,
                    variant=signal.variant,
                    metadata={
                        **learning_meta,
                        "ml_decision": "rejected",
                        "original_direction": signal.direction.value,
                    },
                )
            )
        return tuple(output)


# ---------------------------------------------------------------------------
# Live feedback: record after the response, resolve later
# ---------------------------------------------------------------------------
class LiveSignalRecorder:
    """Persist actionable strategy setups shown by the web analysis.

    Called from a FastAPI background task, i.e. after the response has been sent,
    so recording never delays the recommendation. Errors are logged, never raised.
    """

    def __init__(self, database: Database, *, enabled: bool = True) -> None:
        self.repository = LearningRecordRepository(database)
        self.enabled = enabled

    @staticmethod
    def record_id(symbol: str, timeframe: str, signal: StrategySignal) -> str:
        key = "|".join((symbol.upper(), timeframe, signal.strategy_name, signal.variant, signal.timestamp.isoformat(), signal.direction.value))
        return "live-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

    def record(self, symbol: str, observations: Sequence[tuple[str, StrategySignal, Mapping[str, Any]]]) -> int:
        if not self.enabled:
            return 0
        stored = 0
        for timeframe, signal, values in observations:
            try:
                if signal.state != SignalState.SIGNAL or signal.strategy_name not in LEARNED_STRATEGIES:
                    continue
                features = signal_features(signal, values, timeframe)
                if features is None:
                    continue
                record_id = self.record_id(symbol, timeframe, signal)
                if self.repository.get(record_id) is not None:
                    continue  # same setup already recorded (e.g. repeated analysis of one candle)
                probability = signal.metadata.get("ml_probability") if isinstance(signal.metadata, Mapping) else None
                self.repository.create(
                    _record_for_signal(
                        signal,
                        symbol=symbol,
                        timeframe=timeframe,
                        features=features,
                        source_type=LearningSourceType.LIVE_TRADE,
                        record_id=record_id,
                        confidence=float(probability) * 100.0 if isinstance(probability, (int, float)) else 0.0,
                        provenance={"recorded_by": "web_analyze", "feature_set_version": FEATURE_SET_VERSION},
                    )
                )
                stored += 1
            except Exception:  # noqa: BLE001 - feedback capture must never break the web flow
                logger.exception("failed to record live strategy setup")
        return stored


class LiveOutcomeResolver:
    """Label pending LIVE_TRADE records with the bars that followed them."""

    def __init__(
        self,
        database: Database,
        provider: Any,
        *,
        horizon_bars: int = DEFAULT_HORIZON_BARS,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self.repository = LearningRecordRepository(database)
        self.provider = provider
        self.horizon_bars = max(2, int(horizon_bars))
        self.label_engine = OutcomeLabelEngine(backtest_config or BacktestConfig())

    def pending(self) -> list[LearningRecord]:
        records: list[LearningRecord] = []
        offset = 0
        while True:
            page = self.repository.list(
                status=LearningRecordStatus.PENDING_OUTCOME,
                source_type=LearningSourceType.LIVE_TRADE,
                limit=500,
                offset=offset,
            )
            records.extend(item for item in page if item.feature_set_version == FEATURE_SET_VERSION)
            if len(page) < 500:
                return records
            offset += 500

    def resolve(self, now: datetime | None = None) -> dict[str, int]:
        now = now or datetime.now(timezone.utc)
        summary = {"pending": 0, "completed": 0, "invalid": 0, "waiting": 0, "fetch_errors": 0}
        groups: dict[tuple[str, str], list[LearningRecord]] = {}
        for record in self.pending():
            groups.setdefault((record.symbol, record.timeframe), []).append(record)
            summary["pending"] += 1
        for (symbol, timeframe), records in groups.items():
            minutes = TIMEFRAME_MINUTES.get(timeframe)
            if minutes is None:
                continue
            start = min(item.decision_timestamp for item in records)
            try:
                raw = self.provider.get_ohlc(symbol, timeframe, start, now)
            except Exception as exc:  # noqa: BLE001 - report and continue with other groups
                logger.warning("could not fetch bars to resolve live setups", extra={"symbol": symbol, "timeframe": timeframe, "error": type(exc).__name__})
                summary["fetch_errors"] += 1
                continue
            epoch_minute = int(now.timestamp() // 60)
            forming_bucket = datetime.fromtimestamp((epoch_minute - epoch_minute % minutes) * 60, tz=timezone.utc)
            bars = [
                CanonicalOHLC(timestamp=bar.timestamp, open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=bar.volume)
                for bar in raw
                if bar.timestamp < forming_bucket
            ]
            for record in records:
                future = [bar for bar in bars if bar.timestamp > record.decision_timestamp][: self.horizon_bars]
                complete = len(future) >= self.horizon_bars
                result = self.label_engine.resolve_and_store(self.repository, record.record_id, future, data_complete=complete)
                if result.status == LearningRecordStatus.COMPLETED:
                    summary["completed"] += 1
                elif result.status == LearningRecordStatus.INVALID:
                    summary["invalid"] += 1
                else:
                    summary["waiting"] += 1
        return summary


def live_samples(database: Database) -> list[StrategySample]:
    """Completed LIVE_TRADE records converted into training samples."""
    repository = LearningRecordRepository(database)
    samples: list[StrategySample] = []
    offset = 0
    while True:
        page = repository.list(status=LearningRecordStatus.COMPLETED, source_type=LearningSourceType.LIVE_TRADE, limit=500, offset=offset)
        for record in page:
            if record.feature_set_version != FEATURE_SET_VERSION or record.strategy_name not in LEARNED_STRATEGIES:
                continue
            if record.exit_price is None or record.outcome is None:
                continue
            features = {name: float(record.feature_snapshot.values.get(name, 0.0)) for name in FEATURE_NAMES}
            samples.append(
                StrategySample(
                    strategy=record.strategy_name,
                    symbol=record.symbol,
                    timeframe=record.timeframe,
                    timestamp=record.decision_timestamp,
                    features=features,
                    won=1 if record.outcome == LearningOutcome.TP_BEFORE_SL.value else 0,
                    r_multiple=_r_multiple(record.direction, record.entry_price, record.stop_loss, record.exit_price),
                    source="LIVE",
                )
            )
        if len(page) < 500:
            return samples
        offset += 500


# ---------------------------------------------------------------------------
# End-to-end training run (used by scripts/train_strategies.py)
# ---------------------------------------------------------------------------
def read_ohlc_csv(path: Path) -> list[CanonicalOHLC]:
    """Load one historical CSV with the project's canonical Phase 02 reader."""
    from app.data.loaders.csv_loader import CSVLoader
    from app.data.normalizers.ohlc_normalizer import normalize_row

    unique: dict[datetime, CanonicalOHLC] = {}
    for row in CSVLoader().read_rows(Path(path)):
        bar = normalize_row(row)
        unique.setdefault(bar.timestamp, bar)
    return valid_ohlc(unique[key] for key in sorted(unique))


def collect_historical_samples(
    data_dir: Path,
    builder: HistoricalSampleBuilder,
    *,
    max_bars_per_timeframe: int | None = None,
    progress: Any = None,
) -> tuple[list[StrategySample], dict[str, Any]]:
    """Label every strategy setup found in each ``*.csv`` of ``data_dir``.

    The file name (without extension) is the symbol, e.g. ``BTCUSD.csv``.
    Timeframes finer than a file's native interval are skipped.
    """
    report = progress or (lambda message: None)
    samples: list[StrategySample] = []
    files: dict[str, Any] = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        symbol = path.stem.upper()
        try:
            bars = read_ohlc_csv(path)
        except (OSError, ValueError) as exc:
            files[path.name] = {"error": str(exc)[:200]}
            report(f"  ! {path.name}: skipped ({exc})")
            continue
        entry: dict[str, Any] = {
            "rows": len(bars),
            "native_interval_minutes": base_interval_minutes(bars),
            "first": bars[0].timestamp.isoformat() if bars else None,
            "last": bars[-1].timestamp.isoformat() if bars else None,
            "timeframes": {},
        }
        for timeframe in ANALYSIS_TIMEFRAMES:
            series = timeframe_bars(bars, timeframe)
            if series is None:
                continue
            if max_bars_per_timeframe:
                series = series[-max_bars_per_timeframe:]
            started = time.monotonic()
            found = builder.samples_from_bars(symbol, timeframe, series)
            samples.extend(found)
            entry["timeframes"][timeframe] = {"bars": len(series), "setups": len(found)}
            report(f"  {symbol} {timeframe}: {len(series)} bars -> {len(found)} labelled setups ({time.monotonic() - started:.0f}s)")
        files[path.name] = entry
    return samples, {"files": files}


def run_training(
    *,
    data_dir: Path,
    database: Database,
    model_path: Path,
    provider: Any = None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    max_bars_per_timeframe: int | None = None,
    trainer: StrategyLearningTrainer | None = None,
    progress: Any = None,
) -> StrategyLearningBundle:
    """Resolve live setups, collect historical + live samples, train and save."""
    report = progress or (lambda message: None)
    summary: dict[str, Any] = {"horizon_bars": horizon_bars}
    if provider is not None:
        report("[1/4] Resolving outcomes of recorded live setups ...")
        summary["live_resolution"] = LiveOutcomeResolver(database, provider, horizon_bars=horizon_bars).resolve()
        report(f"  {summary['live_resolution']}")
    else:
        report("[1/4] Live provider not configured: pending live setups stay pending.")
        summary["live_resolution"] = "skipped"

    report(f"[2/4] Learning from historical CSV files in {data_dir} ...")
    builder = HistoricalSampleBuilder(horizon_bars=horizon_bars)
    historical, historical_summary = collect_historical_samples(
        data_dir, builder, max_bars_per_timeframe=max_bars_per_timeframe, progress=report
    )
    summary.update(historical_summary)

    live = live_samples(database)
    summary["historical_samples"] = len(historical)
    summary["live_samples"] = len(live)
    report(f"[3/4] Training per-strategy models on {len(historical)} historical + {len(live)} live setups ...")
    bundle = (trainer or StrategyLearningTrainer()).train([*historical, *live], data_summary=summary)

    bundle.save(model_path)
    report(f"[4/4] Saved models to {model_path}")
    return bundle


def format_report(bundle: StrategyLearningBundle) -> str:
    """Human-readable summary of a training run."""
    lines = [
        "EDGE HUNTER — Strategy machine learning report",
        f"Trained at (UTC): {bundle.trained_at.isoformat()}",
        f"Historical setups: {bundle.data_summary.get('historical_samples')}  |  Live setups: {bundle.data_summary.get('live_samples')}",
        "",
    ]
    for name in LEARNED_STRATEGIES:
        model = bundle.models.get(name)
        if model is None:
            continue
        lines.append(f"{name}: {model.status} — {model.reason}")
        metrics = dict(model.metrics)
        lines.append(f"  samples={metrics.get('samples')} (historical={metrics.get('historical')}, live={metrics.get('live')})")
        oos = metrics.get("oos")
        if isinstance(oos, Mapping):
            lines.append(
                "  out-of-sample: "
                f"all setups avg R={oos.get('baseline_avg_r')} win={oos.get('baseline_win_rate')} (n={oos.get('count')}) | "
                f"with ML filter avg R={oos.get('filtered_avg_r')} win={oos.get('filtered_win_rate')} (kept={oos.get('kept')}), "
                f"threshold={metrics.get('threshold')}"
            )
        lines.append("")
    lines.append("ACTIVE models are applied automatically by the running web app (reloaded within ~30s).")
    lines.append("REJECTED / INSUFFICIENT_DATA strategies keep running exactly as before (no filter).")
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_TIMEFRAMES",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "HistoricalSampleBuilder",
    "LEARNED_STRATEGIES",
    "LiveOutcomeResolver",
    "LiveSignalRecorder",
    "StrategyLearningBundle",
    "StrategyLearningError",
    "StrategyLearningFilter",
    "StrategyLearningTrainer",
    "StrategyModel",
    "StrategySample",
    "live_samples",
    "collect_historical_samples",
    "format_report",
    "read_ohlc_csv",
    "run_training",
    "signal_features",
    "timeframe_bars",
    "valid_ohlc",
]

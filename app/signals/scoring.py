"""Deterministic evidence extraction for strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from app.signals.config import ConfidenceConfig
from app.signals.models import SignalEvidence
from app.strategies.models import SignalDirection, SignalState, StrategySignal


@dataclass(frozen=True)
class EvidenceOverrides:
    """Optional validated evidence supplied by Phase 07 research."""

    oos_performance: float | None = None
    sample_size: float | None = None
    robustness: float | None = None
    market_regime: float | None = None
    entry_quality: float | None = None
    structure_feature_agreement: float | None = None
    stop_target_quality: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "oos_performance": self.oos_performance,
            "sample_size": self.sample_size,
            "robustness": self.robustness,
            "market_regime": self.market_regime,
            "entry_quality": self.entry_quality,
            "structure_feature_agreement": self.structure_feature_agreement,
            "stop_target_quality": self.stop_target_quality,
        }


def score_signal_evidence(
    signal: StrategySignal,
    strategy_agreement: float,
    config: ConfidenceConfig | None = None,
    overrides: EvidenceOverrides | None = None,
) -> tuple[float, SignalEvidence]:
    """Return deterministic local confidence and inspectable components.

    Missing historical/robustness evidence is represented by a neutral 0.5 score
    and explicitly recorded in ``missing_components``; it is never fabricated.
    """
    cfg = config or ConfidenceConfig()
    override = overrides or EvidenceOverrides()
    agreement = _clamp01(strategy_agreement)

    if signal.state != SignalState.SIGNAL:
        neutral = SignalEvidence(
            strategy_agreement=0.0,
            signal_quality=0.0,
            structure_feature_agreement=0.0,
            entry_quality=0.0,
            stop_target_quality=0.0,
            rr_quality=0.0,
            oos_performance=0.0,
            sample_size=0.0,
            robustness=0.0,
            market_regime=0.0,
            missing_components=(),
            sources={},
        )
        return 0.0, neutral

    score_values = [
        float(value)
        for value in signal.score_inputs.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))
    ]
    signal_quality = _clamp01(sum(score_values) / len(score_values)) if score_values else 0.0

    structure = _override_or_metadata(
        override.structure_feature_agreement,
        signal.metadata,
        ("structure_feature_agreement", "structure_score", "feature_agreement"),
    )
    entry = _override_or_metadata(
        override.entry_quality,
        signal.metadata,
        ("entry_quality",),
    )
    stop_target = _override_or_metadata(
        override.stop_target_quality,
        signal.metadata,
        ("stop_target_quality", "sl_target_quality"),
    )
    rr_quality = _rr_quality(signal.risk_reward)

    oos = _override_or_metadata(
        override.oos_performance,
        signal.metadata,
        ("validated_oos_score", "oos_score", "oos_performance"),
    )
    sample = _override_or_metadata(
        override.sample_size,
        signal.metadata,
        ("sample_size_score", "sample_size"),
    )
    robust = _override_or_metadata(
        override.robustness,
        signal.metadata,
        ("robustness_score", "robustness"),
    )
    regime = _override_or_metadata(
        override.market_regime,
        signal.metadata,
        ("regime_score", "market_regime"),
    )

    missing: list[str] = []
    sources: dict[str, str] = {}
    values: dict[str, float] = {
        "strategy_agreement": agreement,
        "signal_quality": signal_quality,
        "structure_feature_agreement": _default_neutral(structure, "structure_feature_agreement", missing, sources),
        "entry_quality": _default_neutral(entry, "entry_quality", missing, sources),
        "stop_target_quality": _default_neutral(stop_target, "stop_target_quality", missing, sources),
        "rr_quality": rr_quality,
        "oos_performance": _default_neutral(oos, "oos_performance", missing, sources),
        "sample_size": _default_neutral(sample, "sample_size", missing, sources),
        "robustness": _default_neutral(robust, "robustness", missing, sources),
        "market_regime": _default_neutral(regime, "market_regime", missing, sources),
    }

    evidence = SignalEvidence(
        strategy_agreement=values["strategy_agreement"],
        signal_quality=values["signal_quality"],
        structure_feature_agreement=values["structure_feature_agreement"],
        entry_quality=values["entry_quality"],
        stop_target_quality=values["stop_target_quality"],
        rr_quality=values["rr_quality"],
        oos_performance=values["oos_performance"],
        sample_size=values["sample_size"],
        robustness=values["robustness"],
        market_regime=values["market_regime"],
        missing_components=tuple(dict.fromkeys(missing)),
        sources=sources,
    )
    weighted = sum(cfg.weights()[name] * values[name] for name in cfg.weights())
    confidence = _clamp01(weighted) * 100.0
    return confidence, evidence


def _override_or_metadata(
    override: float | None,
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    if override is not None:
        return _normalize_external_score(override)
    for key in keys:
        if key in metadata:
            normalized = _normalize_external_score(metadata[key])
            if normalized is not None:
                return normalized
    return None


def _normalize_external_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 0.0 <= number <= 100.0:
        return number / 100.0
    return None


def _default_neutral(
    value: float | None,
    name: str,
    missing: list[str],
    sources: dict[str, str],
) -> float:
    if value is None:
        sources[name] = "neutral_default_due_to_missing_evidence"
        return 0.5
    sources[name] = "validated_override_or_signal_metadata"
    return _clamp01(value)


def _rr_quality(risk_reward: float | None) -> float:
    if risk_reward is None or not isfinite(float(risk_reward)) or risk_reward <= 0:
        return 0.0
    # Engineering normalization: 1R is the zero reference; 2R or above is full.
    return _clamp01((float(risk_reward) - 1.0) / 1.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

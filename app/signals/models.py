"""Canonical API-ready models for signal aggregation and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from app.strategies.models import SignalDirection, SignalState, StrategySignal


class FinalSignalDirection(str, Enum):
    """Final market decision exposed by the signal engine."""

    BUY = "BUY"
    SELL = "SELL"
    NO_CLEAR_SIGNAL = "NO_CLEAR_SIGNAL"


class ConfidenceBand(str, Enum):
    """Configurable human-facing confidence bands."""

    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass(frozen=True)
class SignalEvidence:
    """Inspectable evidence components normalized to the [0, 1] range."""

    strategy_agreement: float
    signal_quality: float
    structure_feature_agreement: float
    entry_quality: float
    stop_target_quality: float
    rr_quality: float
    oos_performance: float
    sample_size: float
    robustness: float
    market_regime: float
    missing_components: tuple[str, ...] = ()
    sources: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | list[str] | dict[str, str]]:
        return {
            "strategy_agreement": self.strategy_agreement,
            "signal_quality": self.signal_quality,
            "structure_feature_agreement": self.structure_feature_agreement,
            "entry_quality": self.entry_quality,
            "stop_target_quality": self.stop_target_quality,
            "rr_quality": self.rr_quality,
            "oos_performance": self.oos_performance,
            "sample_size": self.sample_size,
            "robustness": self.robustness,
            "market_regime": self.market_regime,
            "missing_components": list(self.missing_components),
            "sources": dict(self.sources),
        }


@dataclass(frozen=True)
class StrategyEvaluation:
    """One strategy output plus its local evidence/confidence."""

    strategy_name: str
    variant: str
    state: SignalState
    direction: SignalDirection
    confidence: float
    evidence: SignalEvidence
    signal: StrategySignal

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "variant": self.variant,
            "state": self.state.value,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "evidence": self.evidence.as_dict(),
            "signal": _signal_to_dict(self.signal),
        }


@dataclass(frozen=True)
class FinalSignalDecision:
    """Final single-target decision with all strategy outputs retained."""

    timestamp: datetime
    symbol: str
    timeframe: str
    direction: FinalSignalDirection
    confidence: float
    confidence_label_ar: str
    entry: float | None
    stop_loss: float | None
    target: float | None
    risk_reward: float | None
    selected_strategy: str | None
    selected_variant: str | None
    reasons: tuple[str, ...]
    strategy_evaluations: tuple[StrategyEvaluation, ...]
    directional_support: Mapping[str, float]
    scoring_components: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_trade_signal(self) -> bool:
        return self.direction in (FinalSignalDirection.BUY, FinalSignalDirection.SELL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "confidence_label_ar": self.confidence_label_ar,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "risk_reward": self.risk_reward,
            "selected_strategy": self.selected_strategy,
            "selected_variant": self.selected_variant,
            "reasons": list(self.reasons),
            "strategy_evaluations": [item.to_dict() for item in self.strategy_evaluations],
            "directional_support": dict(self.directional_support),
            "scoring_components": dict(self.scoring_components),
            "metadata": dict(self.metadata),
        }


def _signal_to_dict(signal: StrategySignal) -> dict[str, Any]:
    return {
        "timestamp": signal.timestamp.isoformat(),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "direction": signal.direction.value,
        "state": signal.state.value,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "target": signal.target,
        "risk_reward": signal.risk_reward,
        "entry_logic": signal.entry_logic,
        "invalidation": signal.invalidation,
        "stop_loss_logic": signal.stop_loss_logic,
        "target_logic": signal.target_logic,
        "evidence": list(signal.evidence),
        "score_inputs": dict(signal.score_inputs),
        "strategy_name": signal.strategy_name,
        "variant": signal.variant,
        "metadata": dict(signal.metadata),
    }

"""Configurable selection and confidence-scoring parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceConfig:
    """Controls the decision gates, weights and Arabic confidence labels.

    The defaults are engineering defaults for deterministic aggregation. They are
    not claims of optimal trading performance and should be calibrated later
    against validated research data.
    """

    min_direction_share: float = 0.60
    min_direction_margin: float = 0.20
    min_single_signal_confidence: float = 60.0
    weak_max: float = 39.999
    medium_max: float = 59.999
    strong_max: float = 79.999

    # Weights sum to 1.0. Strategy agreement embeds conflict information.
    w_strategy_agreement: float = 0.20
    w_signal_quality: float = 0.15
    w_structure_feature_agreement: float = 0.10
    w_entry_quality: float = 0.10
    w_stop_target_quality: float = 0.10
    w_rr_quality: float = 0.10
    w_oos_performance: float = 0.10
    w_sample_size: float = 0.05
    w_robustness: float = 0.05
    w_market_regime: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "min_direction_share",
            "min_direction_margin",
            "w_strategy_agreement",
            "w_signal_quality",
            "w_structure_feature_agreement",
            "w_entry_quality",
            "w_stop_target_quality",
            "w_rr_quality",
            "w_oos_performance",
            "w_sample_size",
            "w_robustness",
            "w_market_regime",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_single_signal_confidence < 0.0 or self.min_single_signal_confidence > 100.0:
            raise ValueError("min_single_signal_confidence must be in [0, 100]")
        if not (0.0 < self.weak_max < self.medium_max < self.strong_max <= 100.0):
            raise ValueError("confidence label thresholds must be increasing")
        weights = self.weights()
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("confidence weights must sum to 1.0")

    def weights(self) -> dict[str, float]:
        return {
            "strategy_agreement": self.w_strategy_agreement,
            "signal_quality": self.w_signal_quality,
            "structure_feature_agreement": self.w_structure_feature_agreement,
            "entry_quality": self.w_entry_quality,
            "stop_target_quality": self.w_stop_target_quality,
            "rr_quality": self.w_rr_quality,
            "oos_performance": self.w_oos_performance,
            "sample_size": self.w_sample_size,
            "robustness": self.w_robustness,
            "market_regime": self.w_market_regime,
        }

    def label_for(self, confidence: float) -> str:
        """Return the configured Arabic label for a [0,100] score."""
        value = max(0.0, min(100.0, float(confidence)))
        if value <= self.weak_max:
            return "ضعيف"
        if value <= self.medium_max:
            return "متوسط"
        if value <= self.strong_max:
            return "قوي"
        return "قوي جدًا"

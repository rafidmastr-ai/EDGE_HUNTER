"""Signal selection and confidence engine for EDGE HUNTER Phase 08."""

from app.signals.calibration import (
    CalibrationBin,
    CalibrationReport,
    build_calibration_report,
    monotonicity_violations,
)
from app.signals.config import ConfidenceConfig
from app.signals.engine import SignalConfidenceEngine
from app.signals.models import (
    ConfidenceBand,
    FinalSignalDecision,
    FinalSignalDirection,
    SignalEvidence,
    StrategyEvaluation,
)
from app.signals.scoring import score_signal_evidence
from app.signals.selector import DirectionSelection

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "ConfidenceBand",
    "ConfidenceConfig",
    "DirectionSelection",
    "FinalSignalDecision",
    "FinalSignalDirection",
    "SignalConfidenceEngine",
    "SignalEvidence",
    "StrategyEvaluation",
    "build_calibration_report",
    "monotonicity_violations",
    "score_signal_evidence",
]

"""Optional confidence calibration and monotonicity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration statistics for one confidence interval."""

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    observed_win_rate: float | None
    absolute_gap: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "observed_win_rate": self.observed_win_rate,
            "absolute_gap": self.absolute_gap,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Deterministic calibration report when outcome labels are available."""

    sample_count: int
    brier_score: float | None
    bins: tuple[CalibrationBin, ...]
    monotonicity_violations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "brier_score": self.brier_score,
            "bins": [item.to_dict() for item in self.bins],
            "monotonicity_violations": self.monotonicity_violations,
        }


def build_calibration_report(
    confidences: Sequence[float],
    outcomes: Sequence[int | bool],
    bins: int = 5,
    evidence_scores: Sequence[float] | None = None,
) -> CalibrationReport:
    """Build confidence-vs-outcome calibration diagnostics.

    Confidence is interpreted as a score for diagnostic purposes only. This does
    not transform it into a guaranteed probability.
    """
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have equal length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not confidences:
        return CalibrationReport(0, None, (), 0)
    normalized = [max(0.0, min(100.0, float(value))) for value in confidences]
    labels = [1 if bool(value) else 0 for value in outcomes]
    brier = sum(((confidence / 100.0) - outcome) ** 2 for confidence, outcome in zip(normalized, labels)) / len(normalized)

    step = 100.0 / bins
    report_bins: list[CalibrationBin] = []
    for index in range(bins):
        lower = index * step
        upper = 100.0 if index == bins - 1 else (index + 1) * step
        members = [
            (confidence, outcome)
            for confidence, outcome in zip(normalized, labels)
            if (lower <= confidence < upper) or (index == bins - 1 and lower <= confidence <= upper)
        ]
        if members:
            mean_conf = sum(item[0] for item in members) / len(members)
            observed = sum(item[1] for item in members) / len(members) * 100.0
            gap = abs(mean_conf - observed)
        else:
            mean_conf, observed, gap = None, None, None
        report_bins.append(CalibrationBin(lower, upper, len(members), mean_conf, observed, gap))

    monotonicity_count = 0
    if evidence_scores is not None:
        monotonicity_count = monotonicity_violations(evidence_scores, normalized)
    return CalibrationReport(
        sample_count=len(normalized),
        brier_score=brier,
        bins=tuple(report_bins),
        monotonicity_violations=monotonicity_count,
    )


def monotonicity_violations(evidence_scores: Sequence[float], confidences: Sequence[float]) -> int:
    """Count decreases in confidence after an increase in the controlled evidence score."""
    if len(evidence_scores) != len(confidences):
        raise ValueError("evidence_scores and confidences must have equal length")
    violations = 0
    pairs = list(zip(evidence_scores, confidences))
    for first_evidence, first_confidence in pairs:
        for second_evidence, second_confidence in pairs:
            if second_evidence > first_evidence and second_confidence + 1e-12 < first_confidence:
                violations += 1
    return violations

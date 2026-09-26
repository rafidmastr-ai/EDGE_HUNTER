"""Cross-period, cross-symbol and parameter-perturbation robustness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RobustnessCase:
    name: str
    metrics: Mapping[str, Any]
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "reason": self.reason,
        }


def cross_case_summary(cases: Sequence[RobustnessCase]) -> dict[str, Any]:
    if not cases:
        return {"case_count": 0, "positive_case_ratio": None, "cases": []}
    positive = sum(case.passed for case in cases)
    return {
        "case_count": len(cases),
        "positive_case_ratio": positive / len(cases),
        "cases": [case.to_dict() for case in cases],
    }


def detect_parameter_instability(
    sensitivity_score: float | None,
    worst_neighbor_deterioration: float | None,
    *,
    max_local_deterioration: float = 0.50,
) -> tuple[float | None, tuple[str, ...]]:
    flags: list[str] = []
    if sensitivity_score is None or worst_neighbor_deterioration is None:
        return sensitivity_score, ()
    if worst_neighbor_deterioration > max_local_deterioration:
        flags.append("PARAMETER_SENSITIVE")
    if sensitivity_score < 0.50:
        flags.append("PARAMETER_UNSTABLE")
    return sensitivity_score, tuple(flags)


def summarize_robustness_flags(
    *,
    positive_case_ratio: float | None,
    walk_forward_positive_fold_ratio: float | None,
    sensitivity_flags: Sequence[str] = (),
    minimum_positive_case_ratio: float = 0.50,
    minimum_positive_fold_ratio: float = 0.50,
) -> tuple[str, ...]:
    flags = list(sensitivity_flags)
    if positive_case_ratio is not None and positive_case_ratio < minimum_positive_case_ratio:
        flags.append("CROSS_CASE_UNSTABLE")
    if (
        walk_forward_positive_fold_ratio is not None
        and walk_forward_positive_fold_ratio < minimum_positive_fold_ratio
    ):
        flags.append("WALK_FORWARD_UNSTABLE")
    return tuple(dict.fromkeys(flags))


__all__ = [
    "RobustnessCase",
    "cross_case_summary",
    "detect_parameter_instability",
    "summarize_robustness_flags",
]

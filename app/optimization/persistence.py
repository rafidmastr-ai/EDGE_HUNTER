"""Machine-readable persistence for Phase 07 optimization runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from app.optimization.models import OptimizationEvaluation, OptimizationResult


def save_optimization_result(
    result: OptimizationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "optimization_run.json"
    evaluations_path = destination / "optimization_evaluations.jsonl"
    candidates_path = destination / "validated_candidates.json"
    sensitivity_path = destination / "sensitivity_report.csv"

    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    with evaluations_path.open("w", encoding="utf-8", newline="") as handle:
        for evaluation in result.evaluations:
            handle.write(json.dumps(evaluation.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
    candidates_path.write_text(
        json.dumps(
            [item.to_dict() for item in result.final_candidates],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    rows = list(sensitivity_rows(result.evaluations))
    fields = list(rows[0]) if rows else [
        "evaluation_id", "strategy_name", "worst_neighbor_deterioration", "sensitivity_score", "flags"
    ]
    with sensitivity_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "optimization_run": json_path,
        "evaluations": evaluations_path,
        "validated_candidates": candidates_path,
        "sensitivity": sensitivity_path,
    }


def sensitivity_rows(evaluations: Iterable[OptimizationEvaluation]) -> Iterable[dict[str, object]]:
    for item in evaluations:
        yield {
            "evaluation_id": item.evaluation_id,
            "strategy_name": item.strategy_name,
            "worst_neighbor_deterioration": item.robustness.worst_neighbor_deterioration,
            "sensitivity_score": item.robustness.sensitivity_score,
            "neighbor_count": item.robustness.neighbor_count,
            "parameter_stability": item.robustness.parameter_stability,
            "positive_case_ratio": item.robustness.positive_case_ratio,
            "walk_forward_positive_fold_ratio": item.robustness.walk_forward_positive_fold_ratio,
            "flags": "|".join(item.robustness.flags),
        }


__all__ = ["save_optimization_result", "sensitivity_rows"]

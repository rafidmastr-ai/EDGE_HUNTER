"""Human-readable Phase 07 optimization and robustness report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.optimization.models import OptimizationEvaluation, OptimizationResult


def build_markdown_report(result: OptimizationResult) -> str:
    lines = [
        "# EDGE HUNTER — Phase 07 Optimization, Validation & Robustness",
        "",
        f"**Engine:** `{result.engine_version}`  ",
        f"**Run ID:** `{result.run_id}`  ",
        f"**Strategy:** `{result.strategy_name}`  ",
        f"**Evaluations:** `{len(result.evaluations)}`  ",
        f"**Validated candidates:** `{len(result.final_candidates)}`",
        "",
        "## Policy",
        "",
        "Optimization is restricted to parameters justified by earlier research. Train, validation and OOS observations are kept separate. Higher return alone is not used as the selection criterion.",
        "",
        "## Search Space",
        "",
    ]
    for parameter in result.space.parameters:
        lines.append(
            f"- `{parameter.name}`: `{list(parameter.values)}` — {parameter.reason}"
        )
    lines.extend([
        "",
        "## Split",
        "",
        f"- Train: bars `[0, {result.split.train_end})`",
        f"- Validation: bars `[{result.split.train_end}, {result.split.validation_end})`",
        f"- OOS: bars `[{result.split.validation_end}, {result.split.total_bars})`",
        "",
        "## Evaluation Summary",
        "",
        _table(result.evaluations),
        "",
        "## Validated Candidates",
        "",
    ])
    if result.final_candidates:
        lines.extend(
            f"- `{item.evaluation_id}` — `{item.strategy_name}` configuration survived validation and robustness gates."
            for item in result.final_candidates
        )
    else:
        lines.append("No configuration passed all configured validation and robustness gates.")

    lines.extend([
        "",
        "## Rejected Configurations",
        "",
    ])
    for item in result.evaluations:
        if not item.final_candidate:
            reasons = "; ".join(item.rejection_reasons) or "did not pass final-candidate gates"
            lines.append(f"- `{item.evaluation_id}` — {reasons}.")
    lines.extend([
        "",
        "## Robustness",
        "",
    ])
    for item in result.evaluations:
        robustness = item.robustness
        lines.append(
            f"- `{item.evaluation_id}`: sensitivity={_fmt(robustness.sensitivity_score)}, "
            f"worst-neighbor-deterioration={_fmt(robustness.worst_neighbor_deterioration)}, "
            f"cross-case-positive-ratio={_fmt(robustness.positive_case_ratio)}, "
            f"walk-forward-positive-ratio={_fmt(robustness.walk_forward_positive_fold_ratio)}, "
            f"flags={', '.join(robustness.flags) or 'NONE'}"
        )
    lines.extend([
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{result.space.seed}`",
        f"- Search method: `{result.space.method}`",
        "- Configuration, split plan and evaluation metadata are serialized with the run.",
        "",
    ])
    return "\n".join(lines)


def save_markdown_report(result: OptimizationResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_markdown_report(result), encoding="utf-8")
    return destination


def _table(evaluations: Iterable[OptimizationEvaluation]) -> str:
    rows = []
    for item in evaluations:
        rows.append({
            "id": item.evaluation_id,
            "config": item.config,
            "train_score": round(item.train_score, 4),
            "validation_score": round(item.validation_score, 4),
            "oos_score": round(item.oos_score, 4),
            "val_trades": item.validation.total_trades,
            "oos_trades": item.oos.total_trades,
            "oos_exp_R": round(item.oos.expectancy_r, 5),
            "oos_pf": item.oos.profit_factor,
            "oos_DD%": item.oos.max_drawdown_pct,
            "candidate": item.final_candidate,
        })
    if not rows:
        return "_No evaluations._"
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        return repr(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


__all__ = ["build_markdown_report", "save_markdown_report"]

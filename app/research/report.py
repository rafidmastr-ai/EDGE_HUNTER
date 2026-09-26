"""Human-readable Phase 06 research report generation."""

from __future__ import annotations

from statistics import median
from pathlib import Path
from typing import Iterable

from app.research.models import ResearchExperiment, ResearchRun


def save_markdown_report(run: ResearchRun, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_markdown_report(run), encoding="utf-8")
    return destination


def build_markdown_report(run: ResearchRun) -> str:
    experiments = tuple(run.experiments)
    strategy_rows = _group_rows(experiments, key=lambda exp: exp.strategy_name)
    timeframe_rows = _group_rows(experiments, key=lambda exp: exp.timeframe)
    candidate_rows = [exp for exp in experiments if exp.assessment.candidate_for_phase07]

    lines = [
        "# EDGE HUNTER — Phase 06 Research Report",
        "",
        f"**Research Engine:** `{run.research_engine_version}`  ",
        f"**Run ID:** `{run.run_id}`  ",
        f"**Generated (UTC):** `{run.generated_at_utc.isoformat()}`  ",
        f"**Experiments:** `{len(experiments)}`  ",
        f"**Phase 07 candidates:** `{len(candidate_rows)}`",
        "",
        "## Research Policy",
        "",
        "This report compares configurations using consistent definitions across in-sample and out-of-sample periods. It does not declare a universal best strategy, and candidate labels are screening outputs for Phase 07 rather than profitability guarantees.",
        "",
        "## Per-Strategy OOS Summary",
        "",
        _markdown_table(strategy_rows),
        "",
        "## Per-Timeframe OOS Summary",
        "",
        _markdown_table(timeframe_rows),
        "",
        "## Experiment Comparison",
        "",
        _comparison_table(experiments),
        "",
        "## Parameter / Rule Notes",
        "",
        *_parameter_notes(run),
        "",
        "## Robustness and Failure Observations",
        "",
        *_failure_observations(experiments),
        "",
        "## Phase 07 Candidates",
        "",
    ]
    if candidate_rows:
        for exp in candidate_rows:
            lines.append(
                f"- `{exp.experiment_id}` — {exp.symbol} / {exp.timeframe} / {exp.strategy_name} / {exp.variant_name}: "
                "eligible under the configured evidence screen."
            )
    else:
        lines.append("No experiment met all configured Phase 07 screening gates.")

    lines.extend([
        "",
        "## Rejected / Non-Candidate Configurations",
        "",
    ])
    for exp in experiments:
        if not exp.assessment.candidate_for_phase07:
            reasons = "; ".join(exp.assessment.reasons) if exp.assessment.reasons else "no candidate gate passed"
            lines.append(f"- `{exp.experiment_id}` — {reasons}.")

    lines.extend([
        "",
        "## Dataset and Research Limitations",
        "",
        "- Results depend on the exact canonical OHLC datasets recorded in the run metadata.",
        "- Missing volume disables volume-derived analysis features where volume is not complete.",
        "- Spread and commission are zero unless they are explicitly supplied in the backtest configuration.",
        "- OOS labels are evidence screens, not guarantees of future performance.",
        "",
        "## Reproducibility",
        "",
        "Run metadata, dataset hashes, strategy parameters, backtest configuration and experiment IDs are stored in the machine-readable outputs.",
    ])
    return "\n".join(lines) + "\n"


def _group_rows(experiments: Iterable[ResearchExperiment], key) -> list[dict[str, object]]:
    groups: dict[str, list[ResearchExperiment]] = {}
    for exp in experiments:
        groups.setdefault(key(exp), []).append(exp)
    rows: list[dict[str, object]] = []
    for name in sorted(groups):
        group = groups[name]
        candidates = sum(exp.assessment.candidate_for_phase07 for exp in group)
        rows.append({
            "group": name,
            "experiments": len(group),
            "oos_candidate_count": candidates,
            "median_oos_trades": median(exp.oos_metrics.total_trades for exp in group),
            "median_oos_win_rate": round(median(exp.oos_metrics.win_rate for exp in group), 4),
            "median_oos_pf": _median_optional([exp.oos_metrics.profit_factor for exp in group]),
            "median_oos_expectancy_r": round(median(exp.oos_metrics.expectancy_r for exp in group), 6),
            "median_oos_dd_pct": _median_optional([exp.oos_metrics.max_drawdown_pct for exp in group]),
            "median_positive_period_ratio": round(median(exp.oos_metrics.positive_period_ratio for exp in group), 4),
        })
    return rows


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _comparison_table(experiments: Iterable[ResearchExperiment]) -> str:
    rows = []
    for exp in sorted(experiments, key=lambda item: (item.symbol, item.timeframe, item.strategy_name, item.variant_name)):
        rows.append({
            "id": exp.experiment_id,
            "symbol": exp.symbol,
            "tf": exp.timeframe,
            "strategy": exp.strategy_name,
            "variant": exp.variant_name,
            "IS trades": exp.is_metrics.total_trades,
            "OOS trades": exp.oos_metrics.total_trades,
            "OOS PF": exp.oos_metrics.profit_factor,
            "OOS Exp(R)": exp.oos_metrics.expectancy_r,
            "OOS DD%": exp.oos_metrics.max_drawdown_pct,
            "OOS Pos. Period": exp.oos_metrics.positive_period_ratio,
            "Flags": ", ".join(exp.assessment.flags),
        })
    return _markdown_table(rows)


def _parameter_notes(run: ResearchRun) -> list[str]:
    lines: list[str] = []
    for variant in run.matrix.variants:
        cfg = dict(variant.strategy_config.__dict__)
        lines.append(f"- `{variant.name}`: {variant.description} Parameters: `{cfg}`")
    return lines


def _failure_observations(experiments: Iterable[ResearchExperiment]) -> list[str]:
    counts: dict[str, int] = {}
    for exp in experiments:
        for flag in exp.assessment.flags:
            counts[flag] = counts.get(flag, 0) + 1
    if not counts:
        return ["No experiment assessments were produced."]
    lines = []
    for flag in sorted(counts):
        lines.append(f"- `{flag}`: {counts[flag]} experiment(s).")
    return lines


def _median_optional(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(median(usable), 6) if usable else None


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


__all__ = ["build_markdown_report", "save_markdown_report"]

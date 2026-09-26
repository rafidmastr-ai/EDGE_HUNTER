"""Machine-readable persistence for Phase 06 research runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from app.research.models import ResearchExperiment, ResearchRun


def save_research_run(run: ResearchRun, output_dir: str | Path) -> dict[str, Path]:
    """Write JSON, JSONL and CSV outputs with deterministic field ordering."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / "research_run.json"
    jsonl_path = destination / "research_ledger.jsonl"
    csv_path = destination / "comparison.csv"
    candidates_path = destination / "phase07_candidates.json"
    rejected_path = destination / "rejected_configurations.json"

    json_path.write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8", newline="") as handle:
        for experiment in run.experiments:
            handle.write(json.dumps(experiment.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")

    rows = list(comparison_rows(run.experiments))
    fieldnames = list(rows[0].keys()) if rows else [
        "experiment_id", "symbol", "timeframe", "strategy_name", "strategy_variant", "variant_name",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidates_path.write_text(
        json.dumps([exp.to_dict() for exp in run.candidates], indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    rejected = [
        {
            "experiment_id": exp.experiment_id,
            "symbol": exp.symbol,
            "timeframe": exp.timeframe,
            "strategy_name": exp.strategy_name,
            "strategy_variant": exp.strategy_variant,
            "variant_name": exp.variant_name,
            "flags": list(exp.assessment.flags),
            "reasons": list(exp.assessment.reasons),
        }
        for exp in run.experiments
        if not exp.assessment.candidate_for_phase07
    ]
    rejected_path.write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "research_run": json_path,
        "ledger": jsonl_path,
        "comparison": csv_path,
        "candidates": candidates_path,
        "rejected": rejected_path,
    }


def comparison_rows(experiments: Iterable[ResearchExperiment]) -> Iterable[dict[str, object]]:
    for exp in sorted(
        experiments,
        key=lambda item: (item.symbol, item.timeframe, item.strategy_name, item.variant_name),
    ):
        yield {
            "experiment_id": exp.experiment_id,
            "symbol": exp.symbol,
            "timeframe": exp.timeframe,
            "strategy_name": exp.strategy_name,
            "strategy_variant": exp.strategy_variant,
            "variant_name": exp.variant_name,
            "is_trades": exp.is_metrics.total_trades,
            "oos_trades": exp.oos_metrics.total_trades,
            "is_win_rate": exp.is_metrics.win_rate,
            "oos_win_rate": exp.oos_metrics.win_rate,
            "is_profit_factor": exp.is_metrics.profit_factor,
            "oos_profit_factor": exp.oos_metrics.profit_factor,
            "is_expectancy_r": exp.is_metrics.expectancy_r,
            "oos_expectancy_r": exp.oos_metrics.expectancy_r,
            "oos_avg_rr": exp.oos_metrics.average_rr,
            "oos_max_drawdown": exp.oos_metrics.max_drawdown,
            "oos_max_drawdown_pct": exp.oos_metrics.max_drawdown_pct,
            "oos_net_return": exp.oos_metrics.net_return,
            "oos_positive_period_ratio": exp.oos_metrics.positive_period_ratio,
            "oos_expiry_rate": exp.oos_metrics.expiry_rate,
            "oos_signal_count": exp.oos_signal_stats.signals_generated,
            "oos_signals_per_1000_bars": exp.oos_signal_stats.signals_per_1000_bars,
            "oos_signal_to_trade_ratio": exp.oos_signal_stats.signal_to_trade_ratio,
            "flags": "|".join(exp.assessment.flags),
            "candidate_for_phase07": exp.assessment.candidate_for_phase07,
        }


__all__ = ["comparison_rows", "save_research_run"]

"""Orchestrator for Phase 07 search, validation and robustness checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.backtest.config import BacktestConfig
from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from app.features.models import MarketAnalysisSeries
from app.research.models import ResearchRun
from app.strategies.models import StrategyConfig

from app.optimization.models import (
    OptimizationCandidate,
    OptimizationConfig,
    OptimizationEvaluation,
    OptimizationResult,
    OptimizationSpace,
    ParameterSpec,
    PeriodMetrics,
    RobustnessSummary,
    SplitPlan,
    utc_now,
)
from app.optimization.objectives import is_gate_eligible, score_evidence, score_period
from app.optimization.persistence import save_optimization_result
from app.optimization.robustness import (
    detect_parameter_instability,
    summarize_robustness_flags,
)
from app.optimization.search import generate_search_configs
from app.optimization.sensitivity import compute_sensitivity
from app.optimization.splits import make_split_plan
from app.optimization.validation import evaluate_three_periods, walk_forward_validate


class OptimizationRunner:
    """Deterministic optimizer for one strategy target and one primary dataset.

    The optimization search uses TRAIN only. Validation is used to admit
    candidates. OOS is reported as held-out evidence and is never used to
    generate the search space.
    """

    VERSION = "phase07-v1"

    def __init__(
        self,
        *,
        backtest_config: BacktestConfig | None = None,
        feature_engine: FeatureEngine | None = None,
        optimization_config: OptimizationConfig | None = None,
    ) -> None:
        self.backtest_config = backtest_config or BacktestConfig()
        self.feature_engine = feature_engine or FeatureEngine()
        self.config = optimization_config or OptimizationConfig()

    def optimize(
        self,
        candidate: OptimizationCandidate,
        bars: Sequence[CanonicalOHLC],
        *,
        symbol: str,
        timeframe: str,
        space: OptimizationSpace,
        cross_cases: Sequence[tuple[str, Sequence[CanonicalOHLC]]] = (),
        output_dir: str | None = None,
    ) -> OptimizationResult:
        records = tuple(bars)
        self._validate_bars(records)
        if len(records) < 30:
            raise ValueError("Phase 07 requires a dataset with at least 30 bars for meaningful split testing")
        analysis = self.feature_engine.compute(records, symbol=symbol, timeframe=timeframe)
        split = make_split_plan(
            len(records),
            train_fraction=self.config.train_fraction,
            validation_fraction=self.config.validation_fraction,
        )
        search_configs = generate_search_configs(space)
        evaluations: list[OptimizationEvaluation] = []

        # Search/fit stage: TRAIN score only.
        train_rows: list[tuple[dict[str, Any], float]] = []
        for raw_config in search_configs:
            merged = dict(candidate.base_config)
            merged.update(raw_config)
            strategy_config = StrategyConfig(**merged)
            train, validation, oos = evaluate_three_periods(
                candidate.strategy_name,
                strategy_config,
                bars=records,
                analysis=analysis,
                backtest_config=self.backtest_config,
                train_end=split.train_end,
                validation_end=split.validation_end,
                symbol=symbol,
                timeframe=timeframe,
            )
            train_score = score_period(train)
            train_rows.append((merged, train_score))

        train_rows.sort(key=lambda item: (-item[1], self._config_identity(item[0])))
        shortlisted = train_rows[: self.config.top_train_pool_size]

        preliminary: list[OptimizationEvaluation] = []
        for config_dict, train_score in shortlisted:
            preliminary.append(
                self._evaluate_config(
                    candidate,
                    config_dict,
                    records,
                    analysis,
                    split,
                    symbol,
                    timeframe,
                    cross_cases=(),
                )
            )

        sensitivity = compute_sensitivity(preliminary, score_attr="validation_score", max_distance=self.config.sensitivity_neighbor_count)
        # Add robustness checks after local sensitivity is known.
        for item in preliminary:
            sens = sensitivity.get(item.evaluation_id, {})
            sensitivity_score = sens.get("sensitivity_score")
            worst_deterioration = sens.get("worst_neighbor_deterioration")
            stable_score, sens_flags = detect_parameter_instability(
                sensitivity_score,
                worst_deterioration,
                max_local_deterioration=self.config.max_local_performance_deterioration,
            )
            cross_summary = self._cross_case_evidence(
                candidate,
                item.config,
                cross_cases,
                symbol=symbol,
                timeframe=timeframe,
            )
            walk = walk_forward_validate(
                candidate.strategy_name,
                StrategyConfig(**dict(item.config)),
                bars=records,
                analysis=analysis,
                backtest_config=self.backtest_config,
                symbol=symbol,
                timeframe=timeframe,
                folds=self.config.walk_forward_folds,
            )
            wf_ratio = walk.positive_fold_ratio if walk.folds else None
            flags = summarize_robustness_flags(
                positive_case_ratio=cross_summary,
                walk_forward_positive_fold_ratio=wf_ratio,
                sensitivity_flags=sens_flags,
            )
            robustness = RobustnessSummary(
                sensitivity_score=stable_score,
                neighbor_count=int(sens.get("neighbor_count", 0)),
                worst_neighbor_deterioration=worst_deterioration,
                parameter_stability=stable_score,
                positive_case_ratio=cross_summary,
                walk_forward_positive_fold_ratio=wf_ratio,
                flags=flags,
            )
            eligible, reasons = is_gate_eligible(
                item.train,
                item.validation,
                item.oos,
                self.config,
                robustness_flags=flags,
            )
            final_candidate = eligible and "PARAMETER_SENSITIVE" not in flags and "WALK_FORWARD_UNSTABLE" not in flags
            preliminary_updated = replace(
                item,
                validation_eligible=eligible,
                final_candidate=final_candidate,
                rejection_reasons=tuple(reasons) if not final_candidate else (),
                robustness=robustness,
                metadata={
                    **dict(item.metadata),
                    "walk_forward": walk.to_dict(),
                    "cross_case_positive_ratio": cross_summary,
                },
            )
            evaluations.append(preliminary_updated)

        final_candidates = tuple(
            sorted(
                (item for item in evaluations if item.final_candidate),
                key=lambda item: (-item.validation_score, -item.oos_score, self._config_identity(item.config)),
            )[: self.config.candidate_pool_size]
        )

        run_id = self._run_id(candidate, records, space, split, evaluations)
        result = OptimizationResult(
            engine_version=self.VERSION,
            run_id=run_id,
            generated_at_utc=utc_now(),
            strategy_name=candidate.strategy_name,
            candidate_reference=candidate.to_dict(),
            space=space,
            config=self.config,
            split=split,
            evaluations=tuple(evaluations),
            final_candidates=final_candidates,
            metadata={
                "dataset_sha256": self._data_hash(records),
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper(),
                "feature_engine_version": analysis.engine_version,
                "search_evaluations": len(search_configs),
                "train_shortlist_size": len(shortlisted),
            },
        )
        if output_dir:
            save_optimization_result(result, output_dir)
        return result

    def _evaluate_config(
        self,
        candidate: OptimizationCandidate,
        config_dict: Mapping[str, Any],
        bars: tuple[CanonicalOHLC, ...],
        analysis: MarketAnalysisSeries,
        split: SplitPlan,
        symbol: str,
        timeframe: str,
        *,
        cross_cases: Sequence[tuple[str, Sequence[CanonicalOHLC]]],
    ) -> OptimizationEvaluation:
        train, validation, oos = evaluate_three_periods(
            candidate.strategy_name,
            StrategyConfig(**dict(config_dict)),
            bars=bars,
            analysis=analysis,
            backtest_config=self.backtest_config,
            train_end=split.train_end,
            validation_end=split.validation_end,
            symbol=symbol,
            timeframe=timeframe,
        )
        train_score = score_period(train)
        validation_score = score_evidence(validation)
        oos_score = score_evidence(oos)
        evaluation_id = self._evaluation_id(candidate.strategy_name, config_dict, bars)
        eligible, reasons = is_gate_eligible(
            train,
            validation,
            oos,
            self.config,
        )
        return OptimizationEvaluation(
            evaluation_id=evaluation_id,
            strategy_name=candidate.strategy_name,
            config=dict(config_dict),
            train=train,
            validation=validation,
            oos=oos,
            train_score=train_score,
            validation_score=validation_score,
            oos_score=oos_score,
            validation_eligible=eligible,
            final_candidate=False,
            rejection_reasons=reasons,
            robustness=RobustnessSummary(
                sensitivity_score=None,
                neighbor_count=0,
                worst_neighbor_deterioration=None,
                parameter_stability=None,
                positive_case_ratio=None,
                walk_forward_positive_fold_ratio=None,
                flags=(),
            ),
            metadata={
                "base_reference_experiment_id": candidate.reference_experiment_id,
                "research_reason": candidate.research_reason,
                "cross_case_requested": bool(cross_cases),
            },
        )

    def _cross_case_evidence(
        self,
        candidate: OptimizationCandidate,
        config_dict: Mapping[str, Any],
        cross_cases: Sequence[tuple[str, Sequence[CanonicalOHLC]]],
        *,
        symbol: str,
        timeframe: str,
    ) -> float | None:
        if not cross_cases:
            return None
        config = StrategyConfig(**dict(config_dict))
        passed = 0
        checked = 0
        for case_name, case_bars in cross_cases:
            records = tuple(case_bars)
            if len(records) < 30:
                continue
            analysis = self.feature_engine.compute(records, symbol=case_name, timeframe=timeframe)
            split = make_split_plan(
                len(records),
                train_fraction=self.config.train_fraction,
                validation_fraction=self.config.validation_fraction,
            )
            _, _, oos = evaluate_three_periods(
                candidate.strategy_name,
                config,
                bars=records,
                analysis=analysis,
                backtest_config=self.backtest_config,
                train_end=split.train_end,
                validation_end=split.validation_end,
                symbol=case_name,
                timeframe=timeframe,
            )
            checked += 1
            passed += (
                oos.total_trades >= self.config.min_oos_trades
                and oos.expectancy_r >= self.config.min_oos_expectancy_r
            )
        return passed / checked if checked else None

    @staticmethod
    def default_space() -> OptimizationSpace:
        """Research-justified space derived from Phase 06 tested dimensions."""
        return OptimizationSpace(
            method="grid",
            max_trials=36,
            seed=20260924,
            parameters=(
                ParameterSpec(
                    "min_rr",
                    (1.5, 1.75, 2.0),
                    "Phase 06 explicitly compared fixed 1.50, 1.75 and 2.00 R:R values.",
                ),
                ParameterSpec(
                    "max_rr",
                    (1.5, 1.75, 2.0),
                    "Phase 06 explicitly compared fixed 1.50, 1.75 and 2.00 R:R values and a 1.50–2.00 dynamic range.",
                ),
                ParameterSpec(
                    "minimum_body_ratio",
                    (0.55, 0.65),
                    "Phase 06 tested the baseline 0.55 and stricter 0.65 candle-body confirmation.",
                ),
                ParameterSpec(
                    "swing_lookback",
                    (5, 8),
                    "Phase 06 tested the baseline 5-bar and longer 8-bar structural lookback.",
                ),
            ),
        )

    @staticmethod
    def _validate_bars(bars: Sequence[CanonicalOHLC]) -> None:
        previous = None
        for index, bar in enumerate(bars):
            if previous is not None and bar.timestamp <= previous:
                raise ValueError(f"optimization requires strictly increasing timestamps at index {index}")
            previous = bar.timestamp

    @staticmethod
    def _data_hash(bars: Sequence[CanonicalOHLC]) -> str:
        payload = [
            [
                bar.timestamp.isoformat(),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume) if bar.volume is not None else None,
            ]
            for bar in bars
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _config_identity(config: Mapping[str, Any]) -> str:
        return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def _evaluation_id(
        cls,
        strategy_name: str,
        config: Mapping[str, Any],
        bars: Sequence[CanonicalOHLC],
    ) -> str:
        payload = {
            "strategy": strategy_name,
            "config": dict(config),
            "data_sha256": cls._data_hash(bars),
            "engine_version": cls.VERSION,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

    @classmethod
    def _run_id(
        cls,
        candidate: OptimizationCandidate,
        bars: Sequence[CanonicalOHLC],
        space: OptimizationSpace,
        split: SplitPlan,
        evaluations: Sequence[OptimizationEvaluation],
    ) -> str:
        payload = {
            "candidate": candidate.to_dict(),
            "dataset_sha256": cls._data_hash(bars),
            "space": space.to_dict(),
            "split": split.to_dict(),
            "evaluation_ids": [item.evaluation_id for item in evaluations],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]


def targets_from_research_run(run: ResearchRun) -> tuple[OptimizationCandidate, ...]:
    """Convert Phase 06 candidates into Phase 07 optimization targets."""
    return tuple(
        OptimizationCandidate(
            strategy_name=exp.strategy_name,
            base_config=exp.variant_config,
            reference_experiment_id=exp.experiment_id,
            research_reason=(
                "Admitted by Phase 06 multi-metric evidence screen; "
                "parameters are eligible for documented robustness/optimization testing."
            ),
        )
        for exp in run.candidates
    )


__all__ = ["OptimizationRunner", "targets_from_research_run"]

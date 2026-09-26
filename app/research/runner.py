"""Systematic research runner built on the Phase 05 backtest engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.backtest.config import BacktestConfig
from app.backtest.engine import BacktestEngine
from app.backtest.models import BacktestResult
from app.data.schema import CanonicalOHLC
from app.features.engine import FeatureEngine
from app.features.models import MarketAnalysisSeries
from app.research.assessment import assess_experiment
from app.research.models import (
    MetricSummary,
    ResearchExperiment,
    ResearchMatrix,
    ResearchRun,
    ResearchScreeningConfig,
    ResearchVariant,
    SignalStats,
)
from app.research.resampling import resample_ohlc
from app.research.regimes import summarize_trade_regimes
from app.strategies.models import (
    SignalDirection,
    SignalState,
    Strategy,
    StrategyContext,
    StrategyConfig,
    StrategySignal,
)
from app.strategies.registry import StrategyRegistry


@dataclass
class _Trace:
    evaluations: int = 0
    signals: int = 0
    no_signals: int = 0
    conflicts: int = 0
    insufficient: int = 0

    def record(self, signal: StrategySignal) -> None:
        self.evaluations += 1
        if signal.state == SignalState.SIGNAL:
            self.signals += 1
        elif signal.state == SignalState.CONFLICT:
            self.conflicts += 1
        elif signal.state == SignalState.INSUFFICIENT_DATA:
            self.insufficient += 1
        else:
            self.no_signals += 1


class ResearchRunner:
    """Run a deterministic research matrix over canonical OHLC datasets."""

    VERSION = "phase06-v1"

    def __init__(
        self,
        *,
        backtest_config: BacktestConfig | None = None,
        screening: ResearchScreeningConfig | None = None,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        self.backtest_config = backtest_config or BacktestConfig()
        self.screening = screening or ResearchScreeningConfig()
        self.feature_engine = feature_engine or FeatureEngine()
        self.backtest_engine = BacktestEngine(self.backtest_config)

    def run_dataset(
        self,
        datasets: Mapping[str, Sequence[CanonicalOHLC]],
        matrix: ResearchMatrix,
        *,
        sources: Mapping[str, str] | None = None,
    ) -> ResearchRun:
        sources = sources or {}
        experiments: list[ResearchExperiment] = []
        dataset_metadata: dict[str, Any] = {
            "sources": dict(sources),
            "symbols": {},
            "research_engine_version": self.VERSION,
            "backtest_engine_version": self.backtest_engine.VERSION,
            "analysis_engine_version": self.feature_engine.VERSION,
        }

        for symbol in matrix.symbols:
            if symbol not in datasets:
                continue
            base_bars = tuple(datasets[symbol])
            self._validate_bars(base_bars)
            dataset_metadata["symbols"][symbol] = self._profile_dataset(base_bars)

            for timeframe in matrix.timeframes:
                bars = resample_ohlc(base_bars, timeframe)
                if not bars:
                    continue
                analysis = self.feature_engine.compute(bars, symbol=symbol, timeframe=timeframe)
                for strategy_name in matrix.strategy_names:
                    for variant in matrix.variants:
                        strategy = self._strategy(strategy_name, variant.strategy_config)
                        experiment = self._run_experiment(
                            strategy,
                            bars,
                            analysis,
                            symbol,
                            timeframe,
                            variant,
                            matrix,
                        )
                        experiments.append(experiment)

        run_id = self._run_id(matrix, dataset_metadata, experiments)
        generated = datetime.now(timezone.utc)
        return ResearchRun(
            research_engine_version=self.VERSION,
            run_id=run_id,
            generated_at_utc=generated,
            matrix=matrix,
            screening=self.screening,
            experiments=tuple(experiments),
            dataset_metadata=dataset_metadata,
        )

    def _run_experiment(
        self,
        strategy: Strategy,
        bars: tuple[CanonicalOHLC, ...],
        analysis: MarketAnalysisSeries,
        symbol: str,
        timeframe: str,
        variant: ResearchVariant,
        matrix: ResearchMatrix,
    ) -> ResearchExperiment:
        split_index = self._split_index(len(bars), matrix.is_fraction)
        is_bars = bars[: split_index + 1]
        is_analysis = self._slice_analysis(analysis, 0, split_index + 1)
        is_trace = _Trace()
        is_strategy = _TracingStrategy(strategy, is_trace)
        is_result = self.backtest_engine.run(
            is_strategy,
            is_bars,
            is_analysis,
            symbol=symbol,
            timeframe=timeframe,
        )
        is_metrics = MetricSummary.from_result(is_result)
        is_signal_stats = self._signal_stats(is_trace, len(is_bars), is_result)
        is_regimes = summarize_trade_regimes(is_result.trades, is_analysis)

        oos_trace = _Trace()
        oos_start = bars[split_index + 1].timestamp
        oos_gate = _PeriodGateStrategy(strategy, oos_start, None, oos_trace)
        oos_result = self.backtest_engine.run(
            oos_gate,
            bars,
            analysis,
            symbol=symbol,
            timeframe=timeframe,
        )
        oos_metrics = MetricSummary.from_result(oos_result)
        oos_bar_count = len(bars) - (split_index + 1)
        oos_signal_stats = self._signal_stats(oos_trace, oos_bar_count, oos_result, start=oos_start)
        oos_regimes = summarize_trade_regimes(oos_result.trades, analysis)
        assessment = assess_experiment(
            is_metrics,
            oos_metrics,
            is_signal_stats,
            oos_signal_stats,
            self.screening,
        )

        config_dict = dict(variant.strategy_config.__dict__)
        experiment_id = self._experiment_id(symbol, timeframe, strategy, variant, bars)
        metadata = {
            "dataset_sha256": self._data_hash(bars),
            "split": {
                "is_fraction": matrix.is_fraction,
                "is_end": is_bars[-1].timestamp.isoformat(),
                "oos_start": oos_start.isoformat(),
            },
            "backtest_config": self.backtest_config.to_dict(),
            "strategy_config": config_dict,
            "features": {
                "engine_version": analysis.engine_version,
                "definitions": [definition.name for definition in analysis.definitions],
            },
            "oos_result_summary": self._result_identity(oos_result),
            "is_result_summary": self._result_identity(is_result),
            "regime_summary": {"IS": is_regimes, "OOS": oos_regimes},
            "limitations": self._limitations(bars, analysis),
        }
        return ResearchExperiment(
            experiment_id=experiment_id,
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            strategy_name=strategy.name,
            strategy_variant=strategy.variant,
            variant_name=variant.name,
            variant_config=config_dict,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            is_signal_stats=is_signal_stats,
            oos_signal_stats=oos_signal_stats,
            assessment=assessment,
            metadata=metadata,
        )

    @staticmethod
    def _strategy(name: str, config: StrategyConfig) -> Strategy:
        available = StrategyRegistry.default(config).strategies
        for strategy in available:
            if strategy.name == name:
                return strategy
        raise ValueError(f"strategy not registered: {name}")

    @staticmethod
    def _slice_analysis(
        analysis: MarketAnalysisSeries,
        start: int,
        end: int,
    ) -> MarketAnalysisSeries:
        snapshots = analysis.snapshots[start:end]
        return MarketAnalysisSeries(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            snapshots=snapshots,
            definitions=analysis.definitions,
            warmup_bars_required=analysis.warmup_bars_required,
            volume_features_enabled=analysis.volume_features_enabled,
            engine_version=analysis.engine_version,
        )

    @staticmethod
    def _split_index(length: int, fraction: float) -> int:
        if length < 2:
            raise ValueError("research requires at least two bars")
        index = int(length * fraction) - 1
        return max(0, min(length - 2, index))

    @staticmethod
    def _validate_bars(bars: Sequence[CanonicalOHLC]) -> None:
        previous = None
        for index, bar in enumerate(bars):
            if previous is not None and bar.timestamp <= previous:
                raise ValueError(f"research requires strictly increasing timestamps at index {index}")
            previous = bar.timestamp

    @staticmethod
    def _profile_dataset(bars: Sequence[CanonicalOHLC]) -> dict[str, Any]:
        return {
            "bar_count": len(bars),
            "start": bars[0].timestamp.isoformat() if bars else None,
            "end": bars[-1].timestamp.isoformat() if bars else None,
            "has_volume": bool(bars) and all(bar.volume is not None for bar in bars),
            "data_sha256": ResearchRunner._data_hash(bars),
        }

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
    def _signal_stats(
        trace: _Trace,
        bar_count: int,
        result: BacktestResult,
        *,
        start: datetime | None = None,
    ) -> SignalStats:
        signals_per_1000 = trace.signals / bar_count * 1000.0 if bar_count else 0.0
        trades = result.metrics.total_trades
        ratio = trace.signals / trades if trades else (float("inf") if trace.signals else None)
        return SignalStats(
            evaluations=trace.evaluations,
            signals_generated=trace.signals,
            no_signals=trace.no_signals,
            conflicts=trace.conflicts,
            insufficient_data=trace.insufficient,
            signals_per_1000_bars=signals_per_1000,
            signal_to_trade_ratio=ratio,
        )

    @staticmethod
    def _limitations(bars: Sequence[CanonicalOHLC], analysis: MarketAnalysisSeries) -> tuple[str, ...]:
        limits: list[str] = []
        if not all(bar.volume is not None for bar in bars):
            limits.append("volume data is unavailable or incomplete; volume-derived features are disabled")
        limits.append("spread and commission are zero unless explicitly supplied in backtest configuration")
        if len(bars) < analysis.warmup_bars_required * 2:
            limits.append("dataset is short relative to the feature warm-up horizon")
        return tuple(limits)

    @staticmethod
    def _result_identity(result: BacktestResult) -> dict[str, Any]:
        return {
            "trade_count": len(result.trades),
            "metrics": result.metrics.to_dict(),
            "start_timestamp": result.start_timestamp.isoformat() if result.start_timestamp else None,
            "end_timestamp": result.end_timestamp.isoformat() if result.end_timestamp else None,
        }

    @staticmethod
    def _experiment_id(
        symbol: str,
        timeframe: str,
        strategy: Strategy,
        variant: ResearchVariant,
        bars: Sequence[CanonicalOHLC],
    ) -> str:
        payload = {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "strategy": strategy.name,
            "strategy_variant": strategy.variant,
            "variant": variant.to_dict(),
            "data_sha256": ResearchRunner._data_hash(bars),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @staticmethod
    def _run_id(
        matrix: ResearchMatrix,
        dataset_metadata: Mapping[str, Any],
        experiments: Sequence[ResearchExperiment],
    ) -> str:
        dataset_identity = {
            "symbols": dataset_metadata.get("symbols", {}),
            "research_engine_version": dataset_metadata.get("research_engine_version"),
            "backtest_engine_version": dataset_metadata.get("backtest_engine_version"),
            "analysis_engine_version": dataset_metadata.get("analysis_engine_version"),
        }
        payload = {
            "matrix": matrix.to_dict(),
            "dataset_identity": dataset_identity,
            "experiment_ids": [experiment.experiment_id for experiment in experiments],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]


class _TracingStrategy:
    """Transparent strategy decorator used only for research observability."""

    def __init__(self, strategy: Strategy, trace: _Trace) -> None:
        self._strategy = strategy
        self._trace = trace
        self.name = strategy.name
        self.variant = strategy.variant

    def generate(self, context: StrategyContext) -> StrategySignal:
        signal = self._strategy.generate(context)
        self._trace.record(signal)
        return signal


class _PeriodGateStrategy:
    """Allow a strategy to generate signals only inside the requested OOS window."""

    def __init__(
        self,
        strategy: Strategy,
        start: datetime,
        end: datetime | None,
        trace: _Trace,
    ) -> None:
        self._strategy = strategy
        self._start = start
        self._end = end
        self._trace = trace
        self.name = strategy.name
        self.variant = strategy.variant

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = context.current.timestamp
        if timestamp < self._start or (self._end is not None and timestamp > self._end):
            signal = self._no_signal(context, "outside_research_period")
        else:
            signal = self._strategy.generate(context)
        self._trace.record(signal)
        return signal

    def _no_signal(self, context: StrategyContext, reason: str) -> StrategySignal:
        return StrategySignal(
            timestamp=context.current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            direction=SignalDirection.NO_SIGNAL,
            state=SignalState.NO_SIGNAL,
            entry=None,
            stop_loss=None,
            target=None,
            risk_reward=None,
            entry_logic="No entry generated.",
            invalidation="Outside the configured research period.",
            stop_loss_logic="No stop-loss generated.",
            target_logic="No target generated.",
            evidence=(reason,),
            strategy_name=self.name,
            variant=self.variant,
        )


__all__ = ["ResearchRunner"]

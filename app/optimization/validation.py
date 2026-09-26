"""Out-of-sample, walk-forward and cross-case evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any, Mapping, Sequence

from app.backtest.config import BacktestConfig
from app.backtest.engine import BacktestEngine
from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries
from app.research.models import MetricSummary
from app.strategies.models import SignalDirection, SignalState, Strategy, StrategyConfig, StrategyContext, StrategySignal
from app.strategies.registry import StrategyRegistry

from app.optimization.models import PeriodMetrics, WalkForwardFold, WalkForwardSummary
from app.optimization.splits import make_walk_forward_folds


class _WindowStrategy:
    """Gate an existing strategy into a chronological window without altering context."""

    def __init__(self, inner: Strategy, start: int | None, end: int | None):
        self.inner = inner
        self.start = start
        self.end = end
        self.name = inner.name
        self.variant = inner.variant
        self.last_seen_index = -1

    def generate(self, context: StrategyContext) -> StrategySignal:
        index = context.decision_index
        self.last_seen_index = index
        if self.start is not None and index < self.start:
            return _no_signal(context, "WINDOW_PRE_START")
        if self.end is not None and index >= self.end:
            return _no_signal(context, "WINDOW_POST_END")
        return self.inner.generate(context)


def _no_signal(context: StrategyContext, reason: str) -> StrategySignal:
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
        entry_logic="Window-gated evaluation.",
        invalidation="No active setup.",
        stop_loss_logic="No stop generated.",
        target_logic="No target generated.",
        evidence=(reason,),
        score_inputs={},
        strategy_name="WINDOW_GATE",
        variant="WINDOW_GATE",
        metadata={"window_gate": reason},
    )


def period_metrics(
    *,
    strategy: Strategy,
    bars: Sequence[CanonicalOHLC],
    analysis: MarketAnalysisSeries,
    backtest: BacktestEngine,
    start: int,
    end: int,
    symbol: str,
    timeframe: str,
) -> PeriodMetrics:
    gated = _WindowStrategy(strategy, start, end)
    result = backtest.run(
        gated,
        tuple(bars),
        analysis,
        symbol=symbol,
        timeframe=timeframe,
    )
    return PeriodMetrics.from_metric_summary(MetricSummary.from_result(result))


def evaluate_three_periods(
    strategy_name: str,
    config: StrategyConfig,
    *,
    bars: Sequence[CanonicalOHLC],
    analysis: MarketAnalysisSeries,
    backtest_config: BacktestConfig,
    train_end: int,
    validation_end: int,
    symbol: str,
    timeframe: str,
) -> tuple[PeriodMetrics, PeriodMetrics, PeriodMetrics]:
    strategies = StrategyRegistry.default(config).strategies
    strategy = next((item for item in strategies if item.name == strategy_name), None)
    if strategy is None:
        raise ValueError(f"strategy not registered: {strategy_name}")
    backtest = BacktestEngine(backtest_config)
    train = period_metrics(
        strategy=strategy, bars=bars, analysis=analysis, backtest=backtest,
        start=0, end=train_end, symbol=symbol, timeframe=timeframe,
    )
    validation = period_metrics(
        strategy=strategy, bars=bars, analysis=analysis, backtest=backtest,
        start=train_end, end=validation_end, symbol=symbol, timeframe=timeframe,
    )
    oos = period_metrics(
        strategy=strategy, bars=bars, analysis=analysis, backtest=backtest,
        start=validation_end, end=len(bars), symbol=symbol, timeframe=timeframe,
    )
    return train, validation, oos


def walk_forward_validate(
    strategy_name: str,
    config: StrategyConfig,
    *,
    bars: Sequence[CanonicalOHLC],
    analysis: MarketAnalysisSeries,
    backtest_config: BacktestConfig,
    symbol: str,
    timeframe: str,
    folds: int = 3,
) -> WalkForwardSummary:
    strategy = next(
        (item for item in StrategyRegistry.default(config).strategies if item.name == strategy_name),
        None,
    )
    if strategy is None:
        raise ValueError(f"strategy not registered: {strategy_name}")
    windows = make_walk_forward_folds(len(bars), folds=folds)
    results: list[Mapping[str, Any]] = []
    for fold in windows:
        metrics = period_metrics(
            strategy=strategy,
            bars=bars,
            analysis=analysis,
            backtest=BacktestEngine(backtest_config),
            start=fold.test_start,
            end=fold.test_end,
            symbol=symbol,
            timeframe=timeframe,
        )
        results.append({
            "fold_id": fold.fold_id,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "total_trades": metrics.total_trades,
            "expectancy_r": metrics.expectancy_r,
            "profit_factor": metrics.profit_factor,
            "positive_period_ratio": metrics.positive_period_ratio,
            "win_rate": metrics.win_rate,
        })
    positive = sum(float(item["expectancy_r"]) > 0 for item in results)
    ratio = positive / len(results) if results else 0.0
    pfs = [float(item["profit_factor"]) for item in results if item["profit_factor"] is not None]
    return WalkForwardSummary(
        folds=windows,
        fold_results=tuple(results),
        positive_fold_ratio=ratio,
        mean_expectancy_r=mean(float(item["expectancy_r"]) for item in results) if results else 0.0,
        mean_profit_factor=mean(pfs) if pfs else None,
    )


__all__ = ["evaluate_three_periods", "period_metrics", "walk_forward_validate"]

"""Canonical strategy contracts and signal objects for Phase 04."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from app.data.schema import CanonicalOHLC
from app.features.models import MarketAnalysisSeries, MarketAnalysisSnapshot


class SignalDirection(str, Enum):
    """Direction emitted by a strategy at a decision timestamp."""

    BUY = "BUY"
    SELL = "SELL"
    NO_SIGNAL = "NO_SIGNAL"


class SignalState(str, Enum):
    """Lifecycle-independent state of a strategy evaluation."""

    SIGNAL = "SIGNAL"
    NO_SIGNAL = "NO_SIGNAL"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class StrategyConfig:
    """Common configurable strategy parameters.

    Dynamic R:R is bounded by a research-controlled range. This phase does not
    label any range as optimal; Phase 06/07 can research it later.
    """

    min_rr: float = 1.5
    max_rr: float = 2.0
    swing_lookback: int = 5
    minimum_body_ratio: float = 0.55
    minimum_confirmation_score: float = 2.0

    def __post_init__(self) -> None:
        if self.min_rr <= 0 or self.max_rr <= 0:
            raise ValueError("R:R bounds must be positive")
        if self.min_rr > self.max_rr:
            raise ValueError("min_rr must be <= max_rr")
        if self.swing_lookback < 2:
            raise ValueError("swing_lookback must be >= 2")
        if not 0.0 < self.minimum_body_ratio <= 1.0:
            raise ValueError("minimum_body_ratio must be in (0, 1]")
        if self.minimum_confirmation_score < 0:
            raise ValueError("minimum_confirmation_score must be non-negative")


@dataclass(frozen=True)
class StrategyContext:
    """Read-only decision-time context.

    ``bars`` and ``analysis`` are deliberately sliced to the decision index so a
    strategy cannot accidentally inspect bars/snapshots after its decision time.
    """

    symbol: str
    timeframe: str
    decision_index: int
    bars: tuple[CanonicalOHLC, ...]
    analysis: MarketAnalysisSeries
    current: MarketAnalysisSnapshot

    @classmethod
    def from_series(
        cls,
        bars: Sequence[CanonicalOHLC],
        analysis: MarketAnalysisSeries,
        decision_index: int | None = None,
    ) -> "StrategyContext":
        records = tuple(bars)
        if not records:
            raise ValueError("strategy context requires at least one bar")
        if len(records) != len(analysis.snapshots):
            raise ValueError("bars and analysis snapshots must have equal length")
        index = len(records) - 1 if decision_index is None else decision_index
        if not 0 <= index < len(records):
            raise IndexError("decision_index is outside available history")
        sliced_bars = records[: index + 1]
        sliced_snapshots = analysis.snapshots[: index + 1]
        sliced_analysis = MarketAnalysisSeries(
            symbol=analysis.symbol,
            timeframe=analysis.timeframe,
            snapshots=sliced_snapshots,
            definitions=analysis.definitions,
            warmup_bars_required=analysis.warmup_bars_required,
            volume_features_enabled=analysis.volume_features_enabled,
            engine_version=analysis.engine_version,
        )
        return cls(
            symbol=analysis.symbol.upper(),
            timeframe=analysis.timeframe,
            decision_index=index,
            bars=sliced_bars,
            analysis=sliced_analysis,
            current=sliced_snapshots[-1],
        )


@dataclass(frozen=True)
class StrategySignal:
    """Canonical single-target strategy output."""

    timestamp: datetime
    symbol: str
    timeframe: str
    direction: SignalDirection
    state: SignalState
    entry: float | None
    stop_loss: float | None
    target: float | None
    risk_reward: float | None
    entry_logic: str
    invalidation: str
    stop_loss_logic: str
    target_logic: str
    evidence: tuple[str, ...] = ()
    score_inputs: Mapping[str, float] = field(default_factory=dict)
    strategy_name: str = ""
    variant: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state == SignalState.SIGNAL:
            if self.direction not in (SignalDirection.BUY, SignalDirection.SELL):
                raise ValueError("SIGNAL state requires BUY or SELL direction")
            if self.entry is None or self.stop_loss is None or self.target is None:
                raise ValueError("a valid signal requires entry, stop_loss and one target")
            if self.risk_reward is None or self.risk_reward <= 0:
                raise ValueError("a valid signal requires positive risk_reward")
            if not self.evidence:
                raise ValueError("a valid signal requires evidence")
        else:
            if self.direction != SignalDirection.NO_SIGNAL:
                raise ValueError("non-signal states require NO_SIGNAL direction")
            if self.entry is not None or self.stop_loss is not None or self.target is not None:
                raise ValueError("non-signal outputs must not expose trade prices")
            if self.risk_reward is not None:
                raise ValueError("non-signal outputs must not expose risk_reward")


class Strategy(Protocol):
    """Common interface for all strategy families."""

    name: str
    variant: str

    def generate(self, context: StrategyContext) -> StrategySignal:
        """Evaluate the strategy using only the supplied decision-time context."""
        ...

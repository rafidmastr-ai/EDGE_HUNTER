"""Strategy registry/runner preserving one common interface for all families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from app.strategies.classic import ClassicV1
from app.strategies.ict import ICTV1
from app.strategies.models import Strategy, StrategyConfig, StrategyContext, StrategySignal
from app.strategies.smc import SMCV1


@dataclass(frozen=True)
class StrategyRegistry:
    """Explicitly registered strategy implementations."""

    strategies: tuple[Strategy, ...]

    @classmethod
    def default(cls, config: StrategyConfig | None = None) -> "StrategyRegistry":
        shared = config or StrategyConfig()
        return cls(strategies=(ClassicV1(shared), SMCV1(shared), ICTV1(shared)))

    def names(self) -> tuple[str, ...]:
        return tuple(strategy.name for strategy in self.strategies)

    def variants(self) -> tuple[str, ...]:
        return tuple(strategy.variant for strategy in self.strategies)

    def evaluate_all(self, context: StrategyContext) -> tuple[StrategySignal, ...]:
        """Evaluate every registered strategy on the same decision context."""
        return tuple(strategy.generate(context) for strategy in self.strategies)

    def evaluate_map(self, context: StrategyContext) -> Mapping[str, StrategySignal]:
        """Return signals keyed by strategy name."""
        return {strategy.name: strategy.generate(context) for strategy in self.strategies}


__all__ = [
    "ClassicV1",
    "ICTV1",
    "SMCV1",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "StrategyRegistry",
    "StrategySignal",
]

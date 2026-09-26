"""Deterministic final-direction selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.signals.config import ConfidenceConfig
from app.strategies.models import SignalDirection, SignalState, StrategySignal


@dataclass(frozen=True)
class DirectionSelection:
    """Outcome of the directional evidence gate."""

    direction: SignalDirection
    support: Mapping[str, float]
    reasons: tuple[str, ...]
    selected_signal_indices: tuple[int, ...]

    @property
    def selected(self) -> bool:
        return self.direction in (SignalDirection.BUY, SignalDirection.SELL)


def select_direction(
    signals: Sequence[StrategySignal],
    signal_confidences: Sequence[float],
    config: ConfidenceConfig | None = None,
) -> DirectionSelection:
    """Select BUY/SELL only when directional evidence clears the configured gates."""
    cfg = config or ConfidenceConfig()
    if len(signals) != len(signal_confidences):
        raise ValueError("signals and signal_confidences must have equal length")

    weights: dict[SignalDirection, float] = {
        SignalDirection.BUY: 0.0,
        SignalDirection.SELL: 0.0,
    }
    indices: dict[SignalDirection, list[int]] = {
        SignalDirection.BUY: [],
        SignalDirection.SELL: [],
    }

    for index, (signal, confidence) in enumerate(zip(signals, signal_confidences)):
        if signal.state != SignalState.SIGNAL:
            continue
        direction = signal.direction
        if direction not in weights:
            continue
        # Strategy votes are equal by default. Confidence is not used to manufacture
        # extra strategy weight; it remains an evidence measure after selection.
        weights[direction] += 1.0
        indices[direction].append(index)

    active = weights[SignalDirection.BUY] + weights[SignalDirection.SELL]
    if active == 0:
        return DirectionSelection(
            direction=SignalDirection.NO_SIGNAL,
            support={"BUY": 0.0, "SELL": 0.0},
            reasons=("no_strategy_generated_an_actionable_signal",),
            selected_signal_indices=(),
        )

    buy_share = weights[SignalDirection.BUY] / active
    sell_share = weights[SignalDirection.SELL] / active
    support = {"BUY": buy_share, "SELL": sell_share}
    if buy_share >= sell_share:
        leader, runner_up = SignalDirection.BUY, SignalDirection.SELL
        leader_share, margin = buy_share, buy_share - sell_share
    else:
        leader, runner_up = SignalDirection.SELL, SignalDirection.BUY
        leader_share, margin = sell_share, sell_share - buy_share

    reasons = [
        f"direction_share:{leader.value}={leader_share:.3f}",
        f"direction_margin:{margin:.3f}",
    ]

    if leader_share < cfg.min_direction_share:
        reasons.append("direction_share_below_threshold")
        return DirectionSelection(
            direction=SignalDirection.NO_SIGNAL,
            support=support,
            reasons=tuple(reasons),
            selected_signal_indices=(),
        )
    if margin < cfg.min_direction_margin:
        reasons.append("direction_margin_below_threshold")
        return DirectionSelection(
            direction=SignalDirection.NO_SIGNAL,
            support=support,
            reasons=tuple(reasons),
            selected_signal_indices=(),
        )

    selected_indices = tuple(indices[leader])
    reasons.append(f"direction_selected:{leader.value}")
    return DirectionSelection(
        direction=leader,
        support=support,
        reasons=tuple(reasons),
        selected_signal_indices=selected_indices,
    )

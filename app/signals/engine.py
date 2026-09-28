"""Signal aggregation and deterministic confidence engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from app.signals.config import ConfidenceConfig
from app.signals.costs import CostGate
from app.signals.models import FinalSignalDecision, FinalSignalDirection, StrategyEvaluation
from app.signals.scoring import EvidenceOverrides, score_signal_evidence
from app.signals.selector import select_direction
from app.strategies.models import SignalDirection, SignalState, StrategyContext, StrategySignal
from app.strategies.registry import StrategyRegistry


class SignalFilter(Protocol):
    """Optional post-strategy filter, e.g. the learned per-strategy models.

    It receives every strategy output for one decision context and returns the
    same number of outputs; it may only withhold setups (turn them into
    NO_SIGNAL) or annotate them, never invent new trade prices.
    """

    def apply(self, signals: Sequence[StrategySignal], context: StrategyContext) -> tuple[StrategySignal, ...]:
        ...


@dataclass(frozen=True)
class SignalConfidenceEngine:
    """Run all registered strategies and convert their outputs into one decision."""

    registry: StrategyRegistry
    config: ConfidenceConfig = ConfidenceConfig()
    signal_filter: SignalFilter | None = None
    # Optional cost-aware gate (withholds setups whose cost is too large vs the stop).
    cost_gate: CostGate | None = None

    def evaluate_signals(self, context: StrategyContext) -> tuple[tuple[StrategySignal, ...], tuple[StrategySignal, ...]]:
        """Return (raw strategy outputs, outputs after the optional filter and cost gate)."""
        raw = self.registry.evaluate_all(context)
        filtered = raw
        if self.signal_filter is not None:
            filtered = tuple(self.signal_filter.apply(raw, context))
            self._check_withhold_or_annotate_only(raw, filtered)
        if self.cost_gate is not None:
            filtered = self.cost_gate.apply(filtered)
            self._check_withhold_or_annotate_only(raw, filtered)
        return raw, filtered

    @staticmethod
    def _check_withhold_or_annotate_only(raw: Sequence[StrategySignal], filtered: Sequence[StrategySignal]) -> None:
        if len(filtered) != len(raw) or any(
            item.state == SignalState.SIGNAL
            and (
                original.state != SignalState.SIGNAL
                or (item.direction, item.entry, item.stop_loss, item.target, item.risk_reward)
                != (original.direction, original.entry, original.stop_loss, original.target, original.risk_reward)
            )
            for original, item in zip(raw, filtered)
        ):
            raise ValueError("signal filter may only withhold or annotate strategy outputs")

    def analyze(
        self,
        context: StrategyContext,
        validated_evidence: Mapping[str, EvidenceOverrides] | None = None,
    ) -> FinalSignalDecision:
        """Run every registered strategy using the supplied decision-time context."""
        _, signals = self.evaluate_signals(context)
        return self.decide(context, signals, validated_evidence)

    def decide(
        self,
        context: StrategyContext,
        signals: Sequence[StrategySignal],
        validated_evidence: Mapping[str, EvidenceOverrides] | None = None,
    ) -> FinalSignalDecision:
        """Aggregate already evaluated strategy outputs for one decision context."""
        evidence_map = validated_evidence or {}
        return self.aggregate(
            signals=signals,
            timestamp=context.current.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            validated_evidence=evidence_map,
        )

    def aggregate(
        self,
        signals: Sequence[StrategySignal],
        timestamp,
        symbol: str,
        timeframe: str,
        validated_evidence: Mapping[str, EvidenceOverrides] | None = None,
    ) -> FinalSignalDecision:
        """Aggregate already-computed strategy outputs without accessing market history."""
        ordered = tuple(sorted(signals, key=lambda item: (item.strategy_name, item.variant)))
        self._validate_signal_identity(ordered, timestamp, symbol, timeframe)
        evidence_map = validated_evidence or {}

        active_indices = [
            index for index, signal in enumerate(ordered)
            if signal.state == SignalState.SIGNAL
        ]
        active_count = len(active_indices)
        signal_confidences: list[float] = [0.0] * len(ordered)

        # Agreement is computed from actionable strategy count, not confidence.
        for index in active_indices:
            signal = ordered[index]
            same_direction = sum(
                1
                for other_index in active_indices
                if ordered[other_index].direction == signal.direction
            )
            agreement = same_direction / active_count if active_count else 0.0
            key = f"{signal.strategy_name}:{signal.variant}"
            local_conf, _ = score_signal_evidence(
                signal,
                strategy_agreement=agreement,
                config=self.config,
                overrides=evidence_map.get(key) or evidence_map.get(signal.strategy_name),
            )
            signal_confidences[index] = local_conf

        selection = select_direction(ordered, signal_confidences, self.config)
        evaluations: list[StrategyEvaluation] = []
        for index, signal in enumerate(ordered):
            agreement = (
                sum(
                    1
                    for other_index in active_indices
                    if ordered[other_index].direction == signal.direction
                )
                / active_count
                if signal.state == SignalState.SIGNAL and active_count
                else 0.0
            )
            key = f"{signal.strategy_name}:{signal.variant}"
            local_conf, evidence = score_signal_evidence(
                signal,
                strategy_agreement=agreement,
                config=self.config,
                overrides=evidence_map.get(key) or evidence_map.get(signal.strategy_name),
            )
            evaluations.append(
                StrategyEvaluation(
                    strategy_name=signal.strategy_name,
                    variant=signal.variant,
                    state=signal.state,
                    direction=signal.direction,
                    confidence=local_conf,
                    evidence=evidence,
                    signal=signal,
                )
            )

        final_direction = _to_final_direction(selection.direction)
        if not selection.selected:
            label = self.config.label_for(0.0)
            reasons = tuple(selection.reasons) + self._non_selection_reasons(evaluations)
            return FinalSignalDecision(
                timestamp=timestamp,
                symbol=symbol.upper(),
                timeframe=timeframe,
                direction=final_direction,
                confidence=0.0,
                confidence_label_ar=label,
                entry=None,
                stop_loss=None,
                target=None,
                risk_reward=None,
                selected_strategy=None,
                selected_variant=None,
                reasons=reasons,
                strategy_evaluations=tuple(evaluations),
                directional_support=dict(selection.support),
                scoring_components={},
                metadata={
                    "confidence_is_not_win_rate_or_guaranteed_probability": True,
                    "weak_signals_visible": True,
                    "active_strategy_count": active_count,
                },
            )

        selected_evaluations = [
            evaluations[index] for index in selection.selected_signal_indices
        ]
        final_confidence = sum(item.confidence for item in selected_evaluations) / len(selected_evaluations)

        # A single weak strategy is visible but cannot become a final trade decision.
        if active_count == 1 and final_confidence < self.config.min_single_signal_confidence:
            reasons = (
                *selection.reasons,
                f"single_signal_confidence_below_threshold:{final_confidence:.2f}",
                "weak_signal_visible_but_not_selected",
            )
            return FinalSignalDecision(
                timestamp=timestamp,
                symbol=symbol.upper(),
                timeframe=timeframe,
                direction=FinalSignalDirection.NO_CLEAR_SIGNAL,
                confidence=0.0,
                confidence_label_ar=self.config.label_for(0.0),
                entry=None,
                stop_loss=None,
                target=None,
                risk_reward=None,
                selected_strategy=None,
                selected_variant=None,
                reasons=reasons,
                strategy_evaluations=tuple(evaluations),
                directional_support=dict(selection.support),
                scoring_components={},
                metadata={
                    "confidence_is_not_win_rate_or_guaranteed_probability": True,
                    "weak_signals_visible": True,
                    "active_strategy_count": active_count,
                },
            )

        representative = max(
            selected_evaluations,
            key=lambda item: (item.confidence, item.evidence.strategy_agreement, item.strategy_name, item.variant),
        )
        component_names = (
            "strategy_agreement",
            "signal_quality",
            "structure_feature_agreement",
            "entry_quality",
            "stop_target_quality",
            "rr_quality",
            "oos_performance",
            "sample_size",
            "robustness",
            "market_regime",
        )
        scoring_components = {
            name: round(
                sum(float(getattr(item.evidence, name)) for item in selected_evaluations)
                / len(selected_evaluations),
                6,
            )
            for name in component_names
        }
        reasons = tuple(selection.reasons) + (
            f"representative_strategy:{representative.strategy_name}:{representative.variant}",
            f"final_confidence:{final_confidence:.2f}",
        )
        if representative.evidence.missing_components:
            reasons += (
                "some_validated_evidence_components_are_missing_and_use_neutral_defaults",
            )

        return FinalSignalDecision(
            timestamp=timestamp,
            symbol=symbol.upper(),
            timeframe=timeframe,
            direction=final_direction,
            confidence=round(final_confidence, 6),
            confidence_label_ar=self.config.label_for(final_confidence),
            entry=representative.signal.entry,
            stop_loss=representative.signal.stop_loss,
            target=representative.signal.target,
            risk_reward=representative.signal.risk_reward,
            selected_strategy=representative.strategy_name,
            selected_variant=representative.variant,
            reasons=reasons,
            strategy_evaluations=tuple(evaluations),
            directional_support=dict(selection.support),
            scoring_components={key: value for key, value in scoring_components.items() if isinstance(value, float)},
            metadata={
                "confidence_is_not_win_rate_or_guaranteed_probability": True,
                "weak_signals_visible": True,
                "active_strategy_count": active_count,
                "one_final_target": True,
                "target_source": f"{representative.strategy_name}:{representative.variant}",
            },
        )

    @staticmethod
    def _validate_signal_identity(
        signals: Sequence[StrategySignal],
        timestamp,
        symbol: str,
        timeframe: str,
    ) -> None:
        expected_symbol = symbol.upper()
        seen_names: set[tuple[str, str]] = set()
        for signal in signals:
            identity = (signal.strategy_name, signal.variant)
            if identity in seen_names:
                raise ValueError(f"duplicate strategy output: {identity}")
            seen_names.add(identity)
            if signal.timestamp != timestamp:
                raise ValueError("all strategy outputs must use the same decision timestamp")
            if signal.symbol.upper() != expected_symbol:
                raise ValueError("strategy signal symbol does not match analysis context")
            if signal.timeframe != timeframe:
                raise ValueError("strategy signal timeframe does not match analysis context")

    @staticmethod
    def _non_selection_reasons(evaluations: Sequence[StrategyEvaluation]) -> tuple[str, ...]:
        reasons: list[str] = []
        if any(item.state == SignalState.CONFLICT for item in evaluations):
            reasons.append("one_or_more_strategies_reported_conflict")
        if any(item.state == SignalState.INSUFFICIENT_DATA for item in evaluations):
            reasons.append("one_or_more_strategies_have_insufficient_history")
        if not any(item.state == SignalState.SIGNAL for item in evaluations):
            reasons.append("all_strategy_outputs_are_non_actionable")
        return tuple(reasons)


def _to_final_direction(direction: SignalDirection) -> FinalSignalDirection:
    if direction == SignalDirection.BUY:
        return FinalSignalDirection.BUY
    if direction == SignalDirection.SELL:
        return FinalSignalDirection.SELL
    return FinalSignalDirection.NO_CLEAR_SIGNAL

"""Learning Record Builder for EDGE HUNTER L3.

This module converts an already-produced final decision plus a persisted L2
Feature Snapshot into one immutable-at-capture LearningRecord. It does not
calculate features, re-run strategies, score confidence, or attach outcomes.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningSourceType,
)
from app.learning.repository import LearningRecordRepository
from app.signals.models import FinalSignalDecision, FinalSignalDirection, StrategyEvaluation


class LearningRecordBuildError(ValueError):
    """Raised when a decision cannot be converted into a valid learning record."""


class LearningRecordBuilder:
    """Build decision-time learning records without changing analysis behavior."""

    def __init__(self, feature_store: FeatureStore | None = None) -> None:
        self.feature_store = feature_store

    def build(
        self,
        decision: FinalSignalDecision,
        *,
        feature_snapshot: FeatureSnapshot | None = None,
        feature_snapshot_id: str | None = None,
        source_type: LearningSourceType = LearningSourceType.LIVE_TRADE,
        parameters_snapshot: Mapping[str, Any] | None = None,
        strategy_version: str | None = None,
        dataset_version: str | None = None,
        provenance_metadata: Mapping[str, Any] | None = None,
        entry_timestamp: datetime | None = None,
        record_id: str | None = None,
    ) -> LearningRecord:
        """Build one pending-outcome LearningRecord from an existing decision.

        The builder never recalculates a decision. All decision/evidence values
        are copied at build time and the L2 FeatureSnapshot is referenced by id.
        """
        self._validate_decision(decision)
        snapshot = self._resolve_snapshot(feature_snapshot, feature_snapshot_id)
        self._validate_snapshot_alignment(decision, snapshot)

        selected = self._selected_evaluation(decision)
        signal = selected.signal
        resolved_strategy_version, version_source = self._resolve_strategy_version(
            decision,
            signal.metadata,
            strategy_version,
        )
        parameters = self._resolve_parameters(decision, signal.metadata, parameters_snapshot)
        evidence = self._build_evidence_snapshot(decision, selected)
        provenance = self._build_provenance(
            snapshot,
            source_type=source_type,
            version_source=version_source,
            explicit=provenance_metadata,
        )

        resolved_dataset_version = dataset_version or snapshot.dataset_version
        resolved_entry_timestamp = entry_timestamp or decision.timestamp
        if resolved_entry_timestamp.tzinfo is None or resolved_entry_timestamp.utcoffset() is None:
            raise LearningRecordBuildError("entry_timestamp must be timezone-aware")
        if resolved_entry_timestamp < decision.timestamp:
            raise LearningRecordBuildError("entry_timestamp must not precede decision_timestamp")

        direction = self._map_direction(decision.direction)
        stable_record_id = record_id or self._record_id(
            source_type=source_type,
            decision=decision,
            strategy_version=resolved_strategy_version,
        )

        return LearningRecord(
            record_id=stable_record_id,
            source_type=source_type,
            symbol=decision.symbol.upper(),
            timeframe=decision.timeframe,
            decision_timestamp=decision.timestamp.astimezone(timezone.utc),
            entry_timestamp=resolved_entry_timestamp.astimezone(timezone.utc),
            direction=direction,
            entry_price=float(decision.entry),
            target=float(decision.target),
            stop_loss=float(decision.stop_loss),
            risk_reward=float(decision.risk_reward),
            strategy_name=decision.selected_strategy or selected.strategy_name,
            strategy_variant=decision.selected_variant or selected.variant,
            strategy_version=resolved_strategy_version,
            feature_set_version=snapshot.feature_schema_version,
            parameters_snapshot=parameters,
            feature_snapshot=LearningFeatureSnapshot(snapshot.to_dict()["features"]),
            evidence_snapshot=evidence,
            confidence=float(decision.confidence),
            market_regime=snapshot.market_regime,
            feature_snapshot_id=snapshot.snapshot_id,
            outcome=None,
            exit_timestamp=None,
            exit_price=None,
            exit_reason=None,
            duration=None,
            dataset_version=resolved_dataset_version,
            provenance_metadata=provenance,
        )

    def build_and_store(
        self,
        repository: LearningRecordRepository,
        decision: FinalSignalDecision,
        **kwargs: Any,
    ) -> LearningRecord:
        """Build then persist using the existing L1 repository/idempotency rules."""
        record = self.build(decision, **kwargs)
        return repository.create(record)

    def _resolve_snapshot(
        self,
        feature_snapshot: FeatureSnapshot | None,
        feature_snapshot_id: str | None,
    ) -> FeatureSnapshot:
        if feature_snapshot is not None and feature_snapshot_id is not None:
            if feature_snapshot.snapshot_id != feature_snapshot_id:
                raise LearningRecordBuildError("feature_snapshot and feature_snapshot_id do not refer to the same snapshot")

        if feature_snapshot_id is not None:
            if self.feature_store is None:
                raise LearningRecordBuildError("feature_store is required when feature_snapshot_id is supplied")
            loaded = self.feature_store.load(feature_snapshot_id)
            if loaded is None:
                raise LearningRecordBuildError(f"feature snapshot not found: {feature_snapshot_id}")
            return loaded

        if feature_snapshot is None:
            raise LearningRecordBuildError("a persisted FeatureSnapshot or feature_snapshot_id is required")

        if self.feature_store is not None:
            loaded = self.feature_store.load(feature_snapshot.snapshot_id)
            if loaded is None:
                raise LearningRecordBuildError(
                    f"feature snapshot is not persisted: {feature_snapshot.snapshot_id}"
                )
            return loaded
        return feature_snapshot

    @staticmethod
    def _validate_decision(decision: FinalSignalDecision) -> None:
        if not isinstance(decision, FinalSignalDecision):
            raise LearningRecordBuildError("decision must be a FinalSignalDecision")
        if decision.timestamp.tzinfo is None or decision.timestamp.utcoffset() is None:
            raise LearningRecordBuildError("decision timestamp must be timezone-aware")
        if not decision.is_trade_signal:
            raise LearningRecordBuildError("only an actionable BUY/SELL decision can create a learning record")
        required = {
            "selected_strategy": decision.selected_strategy,
            "selected_variant": decision.selected_variant,
            "entry": decision.entry,
            "target": decision.target,
            "stop_loss": decision.stop_loss,
            "risk_reward": decision.risk_reward,
        }
        missing = [name for name, value in required.items() if value is None or (isinstance(value, str) and not value.strip())]
        if missing:
            raise LearningRecordBuildError(f"decision is missing required fields: {', '.join(missing)}")
        if not math.isfinite(float(decision.confidence)) or not 0.0 <= float(decision.confidence) <= 100.0:
            raise LearningRecordBuildError("decision confidence must be finite and in [0, 100]")

    @staticmethod
    def _validate_snapshot_alignment(decision: FinalSignalDecision, snapshot: FeatureSnapshot) -> None:
        decision_timestamp = decision.timestamp.astimezone(timezone.utc)
        if snapshot.timestamp.astimezone(timezone.utc) != decision_timestamp:
            raise LearningRecordBuildError(
                "FeatureSnapshot timestamp does not match decision timestamp"
            )
        if snapshot.symbol.upper() != decision.symbol.upper():
            raise LearningRecordBuildError("FeatureSnapshot symbol does not match decision symbol")
        if snapshot.timeframe != decision.timeframe:
            raise LearningRecordBuildError("FeatureSnapshot timeframe does not match decision timeframe")

    @staticmethod
    def _selected_evaluation(decision: FinalSignalDecision) -> StrategyEvaluation:
        strategy = decision.selected_strategy
        variant = decision.selected_variant
        matches = [
            evaluation
            for evaluation in decision.strategy_evaluations
            if evaluation.strategy_name == strategy and evaluation.variant == variant
        ]
        if len(matches) != 1:
            raise LearningRecordBuildError(
                "selected strategy evaluation is missing or ambiguous in the decision"
            )
        return matches[0]

    @staticmethod
    def _resolve_strategy_version(
        decision: FinalSignalDecision,
        signal_metadata: Mapping[str, Any],
        explicit_version: str | None,
    ) -> tuple[str, str]:
        if explicit_version is not None:
            value = explicit_version.strip()
            if not value:
                raise LearningRecordBuildError("strategy_version must not be empty when provided")
            return value, "explicit_builder_input"

        for key in ("strategy_version", "version"):
            candidate = signal_metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip(), f"signal_metadata:{key}"

        if decision.selected_variant and decision.selected_variant.strip():
            # Current Strategy contract encodes the released family/version in
            # variants such as Classic_V1, SMC_V1 and ICT_V1. Preserve that
            # exact identifier until a dedicated version field is introduced.
            return decision.selected_variant.strip(), "selected_variant_fallback"
        raise LearningRecordBuildError("strategy_version is unavailable")

    @staticmethod
    def _resolve_parameters(
        decision: FinalSignalDecision,
        signal_metadata: Mapping[str, Any],
        explicit_parameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if explicit_parameters is not None:
            return _copy_json_mapping(explicit_parameters, "parameters_snapshot")
        candidate = signal_metadata.get("parameters_snapshot")
        if candidate is None:
            candidate = decision.metadata.get("parameters_snapshot")
        if candidate is None:
            return {}
        if not isinstance(candidate, Mapping):
            raise LearningRecordBuildError("parameters_snapshot must be a mapping")
        return _copy_json_mapping(candidate, "parameters_snapshot")

    @staticmethod
    def _build_evidence_snapshot(
        decision: FinalSignalDecision,
        selected: StrategyEvaluation,
    ) -> dict[str, Any]:
        payload = {
            "final_reasons": list(decision.reasons),
            "scoring_components": dict(decision.scoring_components),
            "directional_support": dict(decision.directional_support),
            "selected_strategy_evaluation": selected.to_dict(),
            "confidence_label_at_decision": decision.confidence_label_ar,
        }
        return _copy_json_mapping(payload, "evidence_snapshot")

    @staticmethod
    def _build_provenance(
        snapshot: FeatureSnapshot,
        *,
        source_type: LearningSourceType,
        version_source: str,
        explicit: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        combined: dict[str, Any] = copy.deepcopy(dict(snapshot.provenance_metadata))
        if explicit is not None:
            combined.update(copy.deepcopy(dict(explicit)))
        combined.setdefault("source_type", source_type.value)
        combined.setdefault("strategy_version_source", version_source)
        if source_type == LearningSourceType.LIVE_TRADE:
            combined.setdefault("event_kind", "LIVE_SIGNAL_OBSERVATION")
        elif source_type == LearningSourceType.HISTORICAL_BACKTEST:
            combined.setdefault("event_kind", "HISTORICAL_BACKTEST_DECISION")
        return _copy_json_mapping(combined, "provenance_metadata")

    @staticmethod
    def _map_direction(direction: FinalSignalDirection) -> LearningDirection:
        if direction == FinalSignalDirection.BUY:
            return LearningDirection.BUY
        if direction == FinalSignalDirection.SELL:
            return LearningDirection.SELL
        raise LearningRecordBuildError(f"unsupported learning direction: {direction.value}")

    @staticmethod
    def _record_id(
        *,
        source_type: LearningSourceType,
        decision: FinalSignalDecision,
        strategy_version: str,
    ) -> str:
        identity = {
            "source_type": source_type.value,
            "symbol": decision.symbol.upper(),
            "timeframe": decision.timeframe,
            "decision_timestamp": decision.timestamp.astimezone(timezone.utc).isoformat(),
            "direction": decision.direction.value,
            "entry": float(decision.entry),
            "target": float(decision.target),
            "stop_loss": float(decision.stop_loss),
            "risk_reward": float(decision.risk_reward),
            "strategy_name": decision.selected_strategy,
            "strategy_variant": decision.selected_variant,
            "strategy_version": strategy_version,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"lr-{uuid.uuid5(uuid.NAMESPACE_URL, 'edge-hunter:learning-record:' + canonical)}"


def _copy_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    copied = copy.deepcopy(dict(value))
    try:
        json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LearningRecordBuildError(f"{field_name} must be JSON-serializable") from exc
    return copied


__all__ = ["LearningRecordBuildError", "LearningRecordBuilder"]

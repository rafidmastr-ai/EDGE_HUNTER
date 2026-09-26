"""Core Learning contracts for EDGE HUNTER Phase L1.

This module deliberately stops at contracts and lifecycle semantics. It contains
no model training, feature-store implementation, live-learning orchestration, or
AI decision logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping


class LearningSourceType(str, Enum):
    """Broad origin of a learning record."""

    HISTORICAL_BACKTEST = "HISTORICAL_BACKTEST"
    LIVE_TRADE = "LIVE_TRADE"


class LearningRecordStatus(str, Enum):
    """Lifecycle of a learning record."""

    PENDING_OUTCOME = "PENDING_OUTCOME"
    COMPLETED = "COMPLETED"
    INVALID = "INVALID"
    EXCLUDED = "EXCLUDED"


class LearningDirection(str, Enum):
    """Actionable direction stored in a learning record."""

    BUY = "BUY"
    SELL = "SELL"


_JSON_SCALAR = (str, int, float, bool, type(None))
_RESERVED_FEATURE_KEYS = {
    "outcome",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "duration",
    "duration_seconds",
}
_SENSITIVE_KEY_PARTS = ("password", "api_key", "secret", "session_token", "access_token")


def _validate_json_value(value: Any, *, path: str, forbid_outcome: bool = False) -> None:
    """Reject non-JSON values and sensitive/outcome keys recursively."""
    if isinstance(value, _JSON_SCALAR):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            lowered = key.lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(f"{path} contains a prohibited sensitive field: {key}")
            if forbid_outcome and lowered in _RESERVED_FEATURE_KEYS:
                raise ValueError(f"{path} must not contain outcome field: {key}")
            _validate_json_value(item, path=f"{path}.{key}", forbid_outcome=forbid_outcome)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", forbid_outcome=forbid_outcome)
        return
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _validate_snapshot(snapshot: Mapping[str, Any], *, field_name: str, forbid_outcome: bool = False) -> None:
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    _validate_json_value(dict(snapshot), path=field_name, forbid_outcome=forbid_outcome)


def _require_aware_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    _require_aware_utc(parsed, field_name)
    return parsed


@dataclass(frozen=True)
class LearningFeatureSnapshot:
    """Feature-only payload captured at decision time.

    Outcome fields are explicitly rejected here so the feature container cannot
    accidentally become a mixed feature/outcome payload.
    """

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_snapshot(self.values, field_name="feature_snapshot", forbid_outcome=True)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class LearningRecord:
    """Serializable learning foundation record.

    The record can represent historical backtest observations or future live
    observations. ``LIVE_TRADE`` does not imply broker execution; a future
    execution kind may be carried in provenance metadata without changing this
    schema.
    """

    record_id: str
    source_type: LearningSourceType
    symbol: str
    timeframe: str
    decision_timestamp: datetime
    entry_timestamp: datetime | None
    direction: LearningDirection
    entry_price: float
    target: float
    stop_loss: float
    risk_reward: float
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    feature_set_version: str
    parameters_snapshot: Mapping[str, Any]
    feature_snapshot: LearningFeatureSnapshot
    evidence_snapshot: Mapping[str, Any]
    confidence: float
    market_regime: str | None
    feature_snapshot_id: str | None = None
    outcome: str | None = None
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    duration: timedelta | None = None
    dataset_version: str | None = None
    label_version: str | None = None
    provenance_metadata: Mapping[str, Any] = field(default_factory=dict)
    status: LearningRecordStatus = LearningRecordStatus.PENDING_OUTCOME
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must not be empty")
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if not self.strategy_variant.strip():
            raise ValueError("strategy_variant must not be empty")
        if not self.strategy_version.strip():
            raise ValueError("strategy_version must not be empty")
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")

        if self.feature_snapshot_id is not None and not self.feature_snapshot_id.strip():
            raise ValueError("feature_snapshot_id must be non-empty when provided")

        _require_aware_utc(self.decision_timestamp, "decision_timestamp")
        _require_aware_utc(self.created_at, "created_at")
        if self.entry_timestamp is not None:
            _require_aware_utc(self.entry_timestamp, "entry_timestamp")
        if self.exit_timestamp is not None:
            _require_aware_utc(self.exit_timestamp, "exit_timestamp")
        if self.exit_timestamp is not None and self.exit_timestamp < self.decision_timestamp:
            raise ValueError("exit_timestamp must not precede decision_timestamp")
        if self.exit_timestamp is not None and self.entry_timestamp is not None and self.exit_timestamp < self.entry_timestamp:
            raise ValueError("exit_timestamp must not precede entry_timestamp")
        if self.entry_timestamp is not None and self.entry_timestamp < self.decision_timestamp:
            raise ValueError("entry_timestamp must not precede decision_timestamp")

        for field_name, value in (
            ("entry_price", self.entry_price),
            ("target", self.target),
            ("stop_loss", self.stop_loss),
            ("risk_reward", self.risk_reward),
            ("confidence", self.confidence),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be a finite number")
        if self.entry_price <= 0 or self.target <= 0 or self.stop_loss <= 0:
            raise ValueError("entry_price, target and stop_loss must be positive")
        if self.risk_reward <= 0:
            raise ValueError("risk_reward must be positive")
        if not 0.0 <= self.confidence <= 100.0:
            raise ValueError("confidence must be in [0, 100]")
        if self.market_regime is not None and not self.market_regime.strip():
            raise ValueError("market_regime must be non-empty when provided")
        if self.outcome is not None and not self.outcome.strip():
            raise ValueError("outcome must be non-empty when provided")
        if self.exit_price is not None and (not math.isfinite(float(self.exit_price)) or self.exit_price <= 0):
            raise ValueError("exit_price must be a positive finite number when provided")
        if self.exit_reason is not None and not self.exit_reason.strip():
            raise ValueError("exit_reason must be non-empty when provided")
        if self.duration is not None and self.duration.total_seconds() < 0:
            raise ValueError("duration must not be negative")
        if self.dataset_version is not None and not self.dataset_version.strip():
            raise ValueError("dataset_version must be non-empty when provided")
        if self.label_version is not None and not self.label_version.strip():
            raise ValueError("label_version must be non-empty when provided")

        _validate_snapshot(self.parameters_snapshot, field_name="parameters_snapshot")
        _validate_snapshot(self.evidence_snapshot, field_name="evidence_snapshot")
        _validate_snapshot(self.provenance_metadata, field_name="provenance_metadata")

        if self.status == LearningRecordStatus.PENDING_OUTCOME:
            if any(
                value is not None
                for value in (self.outcome, self.exit_timestamp, self.exit_price, self.exit_reason, self.duration)
            ):
                raise ValueError("PENDING_OUTCOME records must not contain outcome fields")
        elif self.status == LearningRecordStatus.COMPLETED:
            if self.outcome is None or self.exit_timestamp is None or self.exit_price is None or self.exit_reason is None or self.duration is None:
                raise ValueError("COMPLETED records require complete outcome fields")

    @property
    def training_eligible(self) -> bool:
        """Return whether the lifecycle permits future training use."""
        return self.status == LearningRecordStatus.COMPLETED

    def identity_payload(self) -> dict[str, Any]:
        """Return stable pre-outcome fields used for duplicate protection."""
        return {
            "source_type": self.source_type.value,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "decision_timestamp": _iso(self.decision_timestamp),
            "entry_timestamp": _iso(self.entry_timestamp),
            "direction": self.direction.value,
            "entry_price": float(self.entry_price),
            "target": float(self.target),
            "stop_loss": float(self.stop_loss),
            "risk_reward": float(self.risk_reward),
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "feature_set_version": self.feature_set_version,
            "parameters_snapshot": dict(self.parameters_snapshot),
            "feature_snapshot": self.feature_snapshot.to_dict(),
            "evidence_snapshot": dict(self.evidence_snapshot),
            "confidence": float(self.confidence),
            "market_regime": self.market_regime,
            "dataset_version": self.dataset_version,
            "label_version": self.label_version,
            "provenance_metadata": dict(self.provenance_metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize without flattening features into outcome fields."""
        return {
            "record_id": self.record_id,
            "source_type": self.source_type.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_timestamp": _iso(self.decision_timestamp),
            "entry_timestamp": _iso(self.entry_timestamp),
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "target": self.target,
            "stop_loss": self.stop_loss,
            "risk_reward": self.risk_reward,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "feature_set_version": self.feature_set_version,
            "feature_snapshot_id": self.feature_snapshot_id,
            "parameters_snapshot": dict(self.parameters_snapshot),
            "feature_snapshot": self.feature_snapshot.to_dict(),
            "evidence_snapshot": dict(self.evidence_snapshot),
            "confidence": self.confidence,
            "market_regime": self.market_regime,
            "outcome": self.outcome,
            "exit_timestamp": _iso(self.exit_timestamp),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "duration_seconds": None if self.duration is None else int(self.duration.total_seconds()),
            "dataset_version": self.dataset_version,
            "label_version": self.label_version,
            "provenance_metadata": dict(self.provenance_metadata),
            "status": self.status.value,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearningRecord":
        """Reconstruct a contract from its JSON-compatible representation."""
        duration_seconds = payload.get("duration_seconds")
        return cls(
            record_id=str(payload["record_id"]),
            source_type=LearningSourceType(payload["source_type"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            decision_timestamp=_parse_datetime(payload["decision_timestamp"], "decision_timestamp"),
            entry_timestamp=_parse_datetime(payload.get("entry_timestamp"), "entry_timestamp"),
            direction=LearningDirection(payload["direction"]),
            entry_price=float(payload["entry_price"]),
            target=float(payload["target"]),
            stop_loss=float(payload["stop_loss"]),
            risk_reward=float(payload["risk_reward"]),
            strategy_name=str(payload["strategy_name"]),
            strategy_variant=str(payload["strategy_variant"]),
            strategy_version=str(payload["strategy_version"]),
            feature_set_version=str(payload["feature_set_version"]),
            feature_snapshot_id=payload.get("feature_snapshot_id"),
            parameters_snapshot=payload.get("parameters_snapshot", {}),
            feature_snapshot=LearningFeatureSnapshot(payload.get("feature_snapshot", {})),
            evidence_snapshot=payload.get("evidence_snapshot", {}),
            confidence=float(payload["confidence"]),
            market_regime=payload.get("market_regime"),
            outcome=payload.get("outcome"),
            exit_timestamp=_parse_datetime(payload.get("exit_timestamp"), "exit_timestamp"),
            exit_price=None if payload.get("exit_price") is None else float(payload["exit_price"]),
            exit_reason=payload.get("exit_reason"),
            duration=None if duration_seconds is None else timedelta(seconds=int(duration_seconds)),
            dataset_version=payload.get("dataset_version"),
            label_version=payload.get("label_version"),
            provenance_metadata=payload.get("provenance_metadata", {}),
            status=LearningRecordStatus(payload.get("status", LearningRecordStatus.PENDING_OUTCOME.value)),
            created_at=_parse_datetime(payload["created_at"], "created_at"),
        )

    def canonical_identity_json(self) -> str:
        """Return canonical JSON used by the repository for identity hashing."""
        return json.dumps(self.identity_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

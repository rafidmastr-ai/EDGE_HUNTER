"""Deterministic learning dataset construction and versioning for EDGE HUNTER L5.

L5 only assembles immutable, completed learning records with their persisted
decision-time FeatureSnapshots and final L4 labels. It never calculates
features, strategies, outcomes, or models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.labels import LearningOutcome
from app.learning.models import (
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)
from app.learning.repository import LearningRecordRepository
from app.optimization.splits import make_split_plan


class DatasetBuildError(ValueError):
    """Raised when a dataset cannot be constructed safely."""

    def __init__(self, message: str, *, exclusions: Mapping[str, int] | None = None) -> None:
        super().__init__(message)
        self.exclusion_counts = dict(exclusions or {})


class DatasetSplit(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    OOS = "OOS"


class DatasetExclusionReason(str, Enum):
    PENDING_OUTCOME = "PENDING_OUTCOME"
    INVALID_OUTCOME = "INVALID_OUTCOME"
    EXCLUDED_RECORD = "EXCLUDED_RECORD"
    MISSING_FEATURE_SNAPSHOT = "MISSING_FEATURE_SNAPSHOT"
    FEATURE_RECORD_MISMATCH = "FEATURE_RECORD_MISMATCH"
    OUTCOME_RECORD_MISMATCH = "OUTCOME_RECORD_MISMATCH"
    MISSING_LABEL_VERSION = "MISSING_LABEL_VERSION"
    LABEL_VERSION_MISMATCH = "LABEL_VERSION_MISMATCH"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    MIXED_FEATURE_SCHEMA = "MIXED_FEATURE_SCHEMA"
    INVALID_FEATURE_PAYLOAD = "INVALID_FEATURE_PAYLOAD"


_OUTCOME_FIELDS = {
    "outcome",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "duration",
    "duration_seconds",
    "r_multiple",
    "pnl",
    "pnl_gross",
    "pnl_net",
}
_SENSITIVE_KEY_PARTS = (
    "password",
    "api_key",
    "secret",
    "session_token",
    "access_token",
    "subscription_code",
)
_ALLOWED_FINAL_OUTCOMES = {item.value for item in LearningOutcome if item != LearningOutcome.INVALID}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DatasetBuildError("dataset timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else _utc(value).isoformat()


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DatasetBuildError("dataset payload contains a non-finite float")
        return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _contains_forbidden_feature_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _OUTCOME_FIELDS:
                return str(key)
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                return str(key)
            nested = _contains_forbidden_feature_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _contains_forbidden_feature_key(item)
            if nested is not None:
                return nested
    return None


def _unique_sorted(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(sorted({value.strip() for value in values if value and value.strip()}))


@dataclass(frozen=True)
class DatasetSplitPolicy:
    """Configurable chronological split policy, using the existing L7 split semantics."""

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    version: str = "v1"

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        if not self.version.strip():
            raise ValueError("split policy version must not be empty")

    @property
    def oos_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "oos_fraction": self.oos_fraction,
            "method": "chronological_groups_with_phase07_index_plan",
        }


@dataclass(frozen=True)
class DatasetRow:
    """One learning row reconstructed from immutable persisted sources."""

    record_id: str
    split: DatasetSplit
    decision_timestamp: datetime
    symbol: str
    timeframe: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    direction: str
    entry_price: float
    target: float
    stop_loss: float
    risk_reward: float
    confidence: float
    parameters_snapshot: Mapping[str, Any]
    evidence_snapshot: Mapping[str, Any]
    market_regime: str | None
    feature_snapshot_id: str
    feature_snapshot_hash: str
    feature_schema_version: str
    features: Mapping[str, Any]
    outcome: str
    label_version: str
    exit_timestamp: datetime | None
    exit_price: float | None
    exit_reason: str | None
    duration_seconds: int | None
    source_type: LearningSourceType
    provenance_metadata: Mapping[str, Any]
    row_fingerprint: str = ""

    def __post_init__(self) -> None:
        ts = _utc(self.decision_timestamp)
        object.__setattr__(self, "decision_timestamp", ts)
        if self.exit_timestamp is not None:
            object.__setattr__(self, "exit_timestamp", _utc(self.exit_timestamp))
        if not self.record_id.strip():
            raise DatasetBuildError("dataset row record_id must not be empty")
        if not self.feature_snapshot_id.strip():
            raise DatasetBuildError("dataset row feature_snapshot_id must not be empty")
        if self.outcome not in _ALLOWED_FINAL_OUTCOMES:
            raise DatasetBuildError(f"dataset row contains non-final outcome: {self.outcome}")
        if not self.label_version.strip():
            raise DatasetBuildError("dataset row label_version must not be empty")
        forbidden = _contains_forbidden_feature_key(self.features)
        if forbidden is not None:
            raise DatasetBuildError(f"future/sensitive feature field is not permitted: {forbidden}")
        payload = self._fingerprint_payload()
        computed = _sha256(payload)
        if self.row_fingerprint and self.row_fingerprint != computed:
            raise DatasetBuildError("row_fingerprint does not match row contents")
        object.__setattr__(self, "row_fingerprint", computed)

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "split": self.split.value,
            "decision_timestamp": _iso(self.decision_timestamp),
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "direction": self.direction,
            "entry_price": float(self.entry_price),
            "target": float(self.target),
            "stop_loss": float(self.stop_loss),
            "risk_reward": float(self.risk_reward),
            "confidence": float(self.confidence),
            "parameters_snapshot": self.parameters_snapshot,
            "evidence_snapshot": self.evidence_snapshot,
            "market_regime": self.market_regime,
            "feature_snapshot_id": self.feature_snapshot_id,
            "feature_snapshot_hash": self.feature_snapshot_hash,
            "feature_schema_version": self.feature_schema_version,
            "features": self.features,
            "outcome": self.outcome,
            "label_version": self.label_version,
            "exit_timestamp": _iso(self.exit_timestamp),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "duration_seconds": self.duration_seconds,
            "source_type": self.source_type.value,
            "provenance_metadata": self.provenance_metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._fingerprint_payload()
        payload["row_fingerprint"] = self.row_fingerprint
        return payload


@dataclass(frozen=True)
class DatasetVersion:
    """Immutable dataset manifest describing exactly how one version was built."""

    dataset_version_id: str
    dataset_name: str
    created_at: datetime
    feature_schema_version: str
    label_version: str
    dataset_policy_version: str
    split_policy_version: str
    row_count: int
    train_count: int
    validation_count: int
    oos_count: int
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    fingerprint: str
    total_records: int
    included_records: int
    excluded_records: int
    statistics: Mapping[str, Any] = field(default_factory=dict)
    filter_scope: Mapping[str, Any] = field(default_factory=dict)
    split_policy: Mapping[str, Any] = field(default_factory=dict)
    exclusion_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", _utc(self.created_at))
        if self.start_timestamp is not None:
            object.__setattr__(self, "start_timestamp", _utc(self.start_timestamp))
        if self.end_timestamp is not None:
            object.__setattr__(self, "end_timestamp", _utc(self.end_timestamp))
        for name in (
            "dataset_version_id",
            "dataset_name",
            "feature_schema_version",
            "label_version",
            "dataset_policy_version",
            "split_policy_version",
            "fingerprint",
        ):
            if not getattr(self, name).strip():
                raise DatasetBuildError(f"{name} must not be empty")
        for name in (
            "row_count",
            "train_count",
            "validation_count",
            "oos_count",
            "total_records",
            "included_records",
            "excluded_records",
        ):
            if getattr(self, name) < 0:
                raise DatasetBuildError(f"{name} must be >= 0")
        if self.row_count != self.included_records:
            raise DatasetBuildError("row_count must equal included_records")
        if self.row_count != self.train_count + self.validation_count + self.oos_count:
            raise DatasetBuildError("split counts must equal row_count")
        if self.total_records != self.included_records + self.excluded_records:
            raise DatasetBuildError("total_records must equal included_records + excluded_records")
        if self.start_timestamp is not None and self.end_timestamp is not None and self.end_timestamp < self.start_timestamp:
            raise DatasetBuildError("dataset end_timestamp must not precede start_timestamp")

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "dataset_name": self.dataset_name,
            "created_at": _iso(self.created_at),
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "dataset_policy_version": self.dataset_policy_version,
            "split_policy_version": self.split_policy_version,
            "row_count": self.row_count,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "oos_count": self.oos_count,
            "start_timestamp": _iso(self.start_timestamp),
            "end_timestamp": _iso(self.end_timestamp),
            "fingerprint": self.fingerprint,
            "total_records": self.total_records,
            "included_records": self.included_records,
            "excluded_records": self.excluded_records,
            "statistics": self.statistics,
            "filter_scope": self.filter_scope,
            "split_policy": self.split_policy,
            "exclusion_counts": self.exclusion_counts,
        }


@dataclass(frozen=True)
class DatasetBuildResult:
    version: DatasetVersion
    rows: tuple[DatasetRow, ...]
    exclusions: Mapping[str, int]


class LearningDatasetManager:
    """Build and persist leakage-safe datasets from completed learning records."""

    DATASET_POLICY_VERSION = "v1"

    def __init__(
        self,
        repository: LearningRecordRepository,
        feature_store: FeatureStore,
    ) -> None:
        self.repository = repository
        self.feature_store = feature_store

    def build(
        self,
        *,
        dataset_name: str = "EDGE_HUNTER_LEARNING",
        symbols: Iterable[str] | None = None,
        timeframes: Iterable[str] | None = None,
        strategies: Iterable[str] | None = None,
        strategy_versions: Iterable[str] | None = None,
        source_types: Iterable[LearningSourceType] | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        feature_schema_version: str | None = None,
        label_version: str | None = None,
        split_policy: DatasetSplitPolicy | None = None,
    ) -> DatasetBuildResult:
        if not dataset_name.strip():
            raise DatasetBuildError("dataset_name must not be empty")
        policy = split_policy or DatasetSplitPolicy()
        start = None if start_timestamp is None else _utc(start_timestamp)
        end = None if end_timestamp is None else _utc(end_timestamp)
        if start is not None and end is not None and end < start:
            raise DatasetBuildError("end_timestamp must not precede start_timestamp")

        symbol_scope = _unique_sorted(symbols)
        timeframe_scope = _unique_sorted(timeframes)
        strategy_scope = _unique_sorted(strategies)
        version_scope = _unique_sorted(strategy_versions)
        source_scope = None if source_types is None else tuple(sorted({item.value for item in source_types}))

        filter_scope = {
            "symbols": symbol_scope,
            "timeframes": timeframe_scope,
            "strategies": strategy_scope,
            "strategy_versions": version_scope,
            "source_types": source_scope,
            "start_timestamp": _iso(start),
            "end_timestamp": _iso(end),
        }

        scoped_records = self._load_records()
        scoped_records = [
            record
            for record in scoped_records
            if self._in_scope(
                record,
                symbols=symbol_scope,
                timeframes=timeframe_scope,
                strategies=strategy_scope,
                strategy_versions=version_scope,
                source_types=source_scope,
                start=start,
                end=end,
            )
        ]
        scoped_records.sort(key=lambda item: (_utc(item.decision_timestamp), item.record_id))
        # A repository join/view may surface the same immutable record more than once.
        # Dataset identity is record_id, so de-duplicate before feature/label assembly.
        unique_records: list[LearningRecord] = []
        seen_record_ids: set[str] = set()
        for record in scoped_records:
            if record.record_id in seen_record_ids:
                continue
            seen_record_ids.add(record.record_id)
            unique_records.append(record)
        scoped_records = unique_records

        exclusions: dict[str, int] = {}
        prepared: list[tuple[LearningRecord, FeatureSnapshot]] = []
        inferred_label_versions: set[str] = set()
        inferred_schema_versions: set[str] = set()

        for record in scoped_records:
            snapshot, reason = self._validate_record(record)
            if snapshot is None:
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            if label_version is not None and record.label_version != label_version:
                exclusions[DatasetExclusionReason.LABEL_VERSION_MISMATCH.value] = exclusions.get(
                    DatasetExclusionReason.LABEL_VERSION_MISMATCH.value, 0
                ) + 1
                continue
            if feature_schema_version is not None and snapshot.feature_schema_version != feature_schema_version:
                exclusions[DatasetExclusionReason.FEATURE_SCHEMA_MISMATCH.value] = exclusions.get(
                    DatasetExclusionReason.FEATURE_SCHEMA_MISMATCH.value, 0
                ) + 1
                continue
            inferred_label_versions.add(record.label_version or "")
            inferred_schema_versions.add(snapshot.feature_schema_version)
            prepared.append((record, snapshot))

        if label_version is None:
            nonempty = {item for item in inferred_label_versions if item}
            if len(nonempty) > 1:
                raise DatasetBuildError("MIXED_LABEL_VERSIONS: specify label_version explicitly")
            resolved_label_version = next(iter(nonempty), "")
        else:
            resolved_label_version = label_version.strip()
            if not resolved_label_version:
                raise DatasetBuildError("label_version must not be empty")

        if feature_schema_version is None:
            if len(inferred_schema_versions) > 1:
                raise DatasetBuildError("MIXED_FEATURE_SCHEMA: specify feature_schema_version explicitly")
            resolved_schema_version = next(iter(inferred_schema_versions), "")
        else:
            resolved_schema_version = feature_schema_version.strip()
            if not resolved_schema_version:
                raise DatasetBuildError("feature_schema_version must not be empty")

        if not prepared:
            raise DatasetBuildError("NO_ELIGIBLE_RECORDS", exclusions=exclusions)

        split_assignments = self._assign_splits([record for record, _ in prepared], policy)
        rows: list[DatasetRow] = []
        for index, (record, snapshot) in enumerate(prepared):
            rows.append(
                DatasetRow(
                    record_id=record.record_id,
                    split=split_assignments[index],
                    decision_timestamp=record.decision_timestamp,
                    symbol=record.symbol,
                    timeframe=record.timeframe,
                    strategy_name=record.strategy_name,
                    strategy_variant=record.strategy_variant,
                    strategy_version=record.strategy_version,
                    direction=record.direction.value,
                    entry_price=record.entry_price,
                    target=record.target,
                    stop_loss=record.stop_loss,
                    risk_reward=record.risk_reward,
                    confidence=record.confidence,
                    parameters_snapshot=record.parameters_snapshot,
                    evidence_snapshot=record.evidence_snapshot,
                    market_regime=record.market_regime,
                    feature_snapshot_id=snapshot.snapshot_id,
                    feature_snapshot_hash=snapshot.snapshot_hash,
                    feature_schema_version=snapshot.feature_schema_version,
                    features=snapshot.features,
                    outcome=record.outcome or "",
                    label_version=record.label_version or "",
                    exit_timestamp=record.exit_timestamp,
                    exit_price=record.exit_price,
                    exit_reason=record.exit_reason,
                    duration_seconds=None if record.duration is None else int(record.duration.total_seconds()),
                    source_type=record.source_type,
                    provenance_metadata=record.provenance_metadata,
                )
            )
        rows.sort(key=lambda row: (_utc(row.decision_timestamp), row.record_id))

        self._validate_temporal_splits(rows)
        split_counts = {split: sum(1 for row in rows if row.split == split) for split in DatasetSplit}
        statistics = self._statistics(rows)
        fingerprint_payload = {
            "dataset_name": dataset_name,
            "dataset_policy_version": self.DATASET_POLICY_VERSION,
            "feature_schema_version": resolved_schema_version,
            "label_version": resolved_label_version,
            "filter_scope": filter_scope,
            "split_policy": policy.to_dict(),
            "rows": [row._fingerprint_payload() for row in rows],
            "exclusions": exclusions,
        }
        fingerprint = _sha256(fingerprint_payload)
        version_id = f"dsv1-{fingerprint[:32]}"

        version = DatasetVersion(
            dataset_version_id=version_id,
            dataset_name=dataset_name,
            created_at=datetime.now(timezone.utc),
            feature_schema_version=resolved_schema_version,
            label_version=resolved_label_version,
            dataset_policy_version=self.DATASET_POLICY_VERSION,
            split_policy_version=policy.version,
            row_count=len(rows),
            train_count=split_counts[DatasetSplit.TRAIN],
            validation_count=split_counts[DatasetSplit.VALIDATION],
            oos_count=split_counts[DatasetSplit.OOS],
            start_timestamp=rows[0].decision_timestamp,
            end_timestamp=rows[-1].decision_timestamp,
            fingerprint=fingerprint,
            total_records=len(scoped_records),
            included_records=len(rows),
            excluded_records=len(scoped_records) - len(rows),
            statistics=statistics,
            filter_scope=filter_scope,
            split_policy=policy.to_dict(),
            exclusion_counts=exclusions,
        )
        return DatasetBuildResult(version=version, rows=tuple(rows), exclusions=dict(exclusions))

    def build_and_store(self, **kwargs: Any) -> DatasetBuildResult:
        """Build a dataset and persist its immutable version and row references."""
        result = self.build(**kwargs)
        from app.learning.dataset_repository import DatasetVersionRepository

        DatasetVersionRepository(self.repository.database).create(result.version, result.rows)
        return result

    def _load_records(self) -> list[LearningRecord]:
        all_records: list[LearningRecord] = []
        offset = 0
        while True:
            batch = self.repository.list(limit=500, offset=offset)
            if not batch:
                break
            all_records.extend(batch)
            if len(batch) < 500:
                break
            offset += len(batch)
        return all_records

    def _in_scope(
        self,
        record: LearningRecord,
        *,
        symbols: tuple[str, ...] | None,
        timeframes: tuple[str, ...] | None,
        strategies: tuple[str, ...] | None,
        strategy_versions: tuple[str, ...] | None,
        source_types: tuple[str, ...] | None,
        start: datetime | None,
        end: datetime | None,
    ) -> bool:
        if symbols is not None and record.symbol.upper() not in {item.upper() for item in symbols}:
            return False
        if timeframes is not None and record.timeframe not in timeframes:
            return False
        if strategies is not None and record.strategy_name not in strategies:
            return False
        if strategy_versions is not None and record.strategy_version not in strategy_versions:
            return False
        if source_types is not None and record.source_type.value not in source_types:
            return False
        timestamp = _utc(record.decision_timestamp)
        if start is not None and timestamp < start:
            return False
        if end is not None and timestamp > end:
            return False
        return True

    def _validate_record(self, record: LearningRecord) -> tuple[FeatureSnapshot | None, str]:
        if record.status == LearningRecordStatus.PENDING_OUTCOME:
            return None, DatasetExclusionReason.PENDING_OUTCOME.value
        if record.status == LearningRecordStatus.INVALID or record.outcome == LearningOutcome.INVALID.value:
            return None, DatasetExclusionReason.INVALID_OUTCOME.value
        if record.status != LearningRecordStatus.COMPLETED or not record.training_eligible:
            return None, DatasetExclusionReason.EXCLUDED_RECORD.value
        if not record.label_version:
            return None, DatasetExclusionReason.MISSING_LABEL_VERSION.value
        if record.outcome not in _ALLOWED_FINAL_OUTCOMES:
            return None, DatasetExclusionReason.OUTCOME_RECORD_MISMATCH.value
        if not record.feature_snapshot_id:
            return None, DatasetExclusionReason.MISSING_FEATURE_SNAPSHOT.value
        snapshot = self.feature_store.get(record.feature_snapshot_id)
        if snapshot is None:
            return None, DatasetExclusionReason.MISSING_FEATURE_SNAPSHOT.value
        if snapshot.snapshot_id != record.feature_snapshot_id:
            return None, DatasetExclusionReason.FEATURE_RECORD_MISMATCH.value
        if (
            _utc(snapshot.timestamp) != _utc(record.decision_timestamp)
            or snapshot.symbol.upper() != record.symbol.upper()
            or snapshot.timeframe != record.timeframe
        ):
            return None, DatasetExclusionReason.FEATURE_RECORD_MISMATCH.value
        forbidden = _contains_forbidden_feature_key(snapshot.features)
        if forbidden is not None:
            return None, DatasetExclusionReason.INVALID_FEATURE_PAYLOAD.value
        if record.exit_timestamp is None or record.exit_price is None or record.exit_reason is None or record.duration is None:
            return None, DatasetExclusionReason.OUTCOME_RECORD_MISMATCH.value
        return snapshot, ""

    @staticmethod
    def _assign_splits(records: Sequence[LearningRecord], policy: DatasetSplitPolicy) -> list[DatasetSplit]:
        if not records:
            return []
        groups: list[list[int]] = []
        current_timestamp: datetime | None = None
        for index, record in enumerate(records):
            timestamp = _utc(record.decision_timestamp)
            if current_timestamp != timestamp:
                groups.append([])
                current_timestamp = timestamp
            groups[-1].append(index)

        group_count = len(groups)
        assignments = [DatasetSplit.TRAIN] * len(records)
        if group_count == 1:
            return assignments
        if group_count == 2:
            for index in groups[1]:
                assignments[index] = DatasetSplit.OOS
            return assignments

        if policy.validation_fraction == 0.0:
            train_end = max(1, min(group_count - 1, int(group_count * policy.train_fraction)))
            for group_index in range(train_end, group_count):
                for record_index in groups[group_index]:
                    assignments[record_index] = DatasetSplit.OOS
            return assignments

        split_plan = make_split_plan(
            group_count,
            train_fraction=policy.train_fraction,
            validation_fraction=policy.validation_fraction,
        )
        for group_index in range(split_plan.train_end, split_plan.validation_end):
            for record_index in groups[group_index]:
                assignments[record_index] = DatasetSplit.VALIDATION
        for group_index in range(split_plan.validation_end, group_count):
            for record_index in groups[group_index]:
                assignments[record_index] = DatasetSplit.OOS
        return assignments

    @staticmethod
    def _validate_temporal_splits(rows: Sequence[DatasetRow]) -> None:
        for earlier, later in zip(rows, rows[1:]):
            if _utc(later.decision_timestamp) < _utc(earlier.decision_timestamp):
                raise DatasetBuildError("dataset rows are not temporally ordered")
        split_first: dict[DatasetSplit, datetime | None] = {}
        split_last: dict[DatasetSplit, datetime | None] = {}
        for row in rows:
            timestamp = _utc(row.decision_timestamp)
            split_first.setdefault(row.split, timestamp)
            split_last[row.split] = timestamp
        train_last = split_last.get(DatasetSplit.TRAIN)
        validation_first = split_first.get(DatasetSplit.VALIDATION)
        validation_last = split_last.get(DatasetSplit.VALIDATION)
        oos_first = split_first.get(DatasetSplit.OOS)
        if train_last is not None and validation_first is not None and train_last >= validation_first:
            raise DatasetBuildError("TRAIN and VALIDATION overlap temporally")
        if validation_last is not None and oos_first is not None and validation_last >= oos_first:
            raise DatasetBuildError("VALIDATION and OOS overlap temporally")
        if oos_first is not None and train_last is not None and train_last >= oos_first and validation_first is None:
            raise DatasetBuildError("TRAIN and OOS overlap temporally")

    @staticmethod
    def _statistics(rows: Sequence[DatasetRow]) -> dict[str, Any]:
        def counts(values: Iterable[str]) -> dict[str, int]:
            result: dict[str, int] = {}
            for value in values:
                result[value] = result.get(value, 0) + 1
            return dict(sorted(result.items()))

        return {
            "by_split": counts(row.split.value for row in rows),
            "by_strategy": counts(row.strategy_name for row in rows),
            "by_direction": counts(row.direction for row in rows),
            "by_outcome": counts(row.outcome for row in rows),
            "by_symbol": counts(row.symbol.upper() for row in rows),
            "by_timeframe": counts(row.timeframe for row in rows),
            "by_provenance": counts(row.source_type.value for row in rows),
        }


__all__ = [
    "DatasetBuildError",
    "DatasetBuildResult",
    "DatasetExclusionReason",
    "DatasetRow",
    "DatasetSplit",
    "DatasetSplitPolicy",
    "DatasetVersion",
    "LearningDatasetManager",
]

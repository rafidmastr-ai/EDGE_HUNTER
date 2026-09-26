"""Decision-time Feature Snapshot contracts, canonical serialization and storage.

This module is intentionally limited to persistence foundation concerns. It does
not calculate features, train models, consume live trades, or alter the current
analysis/decision flow.
"""

from __future__ import annotations

import hashlib
import sqlite3
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any

from app.db.database import Database
from app.features.models import MarketAnalysisSnapshot


FEATURE_SCHEMA_VERSION = "phase03-feature-schema-v1"
_SPECIAL_TYPE_KEY = "__edge_hunter_type__"
_OUTCOME_KEYS = {
    "outcome",
    "exit_timestamp",
    "exit_price",
    "exit_reason",
    "duration",
    "duration_seconds",
    "pnl",
    "tp_hit",
    "sl_hit",
}
_SENSITIVE_KEY_PARTS = ("password", "api_key", "secret", "session_token", "access_token")


class FeatureSnapshotError(ValueError):
    """Base error for invalid Feature Snapshot operations."""


class DuplicateFeatureSnapshotError(FeatureSnapshotError):
    """Raised when a snapshot id conflicts with a different snapshot."""


class FeatureSnapshotImmutableError(FeatureSnapshotError):
    """Raised when an existing immutable snapshot would be overwritten."""


def _require_aware_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FeatureSnapshotError(f"{field_name} must be timezone-aware")


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode_datetime(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    _require_aware_utc(parsed, field_name)
    return parsed


def _canonicalize(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation for hashing/storage."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeatureSnapshotError("snapshot contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FeatureSnapshotError("snapshot contains a non-finite Decimal")
        return {_SPECIAL_TYPE_KEY: "decimal", "value": format(value, "f")}
    if isinstance(value, datetime):
        _require_aware_utc(value, "snapshot datetime")
        return {_SPECIAL_TYPE_KEY: "datetime", "value": _utc_iso(value)}
    if isinstance(value, Enum):
        # Persist semantic enum value, not a Python import path. This keeps the
        # stored representation safe and provider/model-version independent.
        return _canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    module = type(value).__module__
    if module.startswith("numpy"):
        if hasattr(value, "tolist"):
            return _canonicalize(value.tolist())
        if hasattr(value, "item"):
            return _canonicalize(value.item())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FeatureSnapshotError("snapshot mappings must use string keys")
            lowered = key.lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                raise FeatureSnapshotError(f"snapshot contains a prohibited sensitive field: {key}")
            if lowered in _OUTCOME_KEYS:
                raise FeatureSnapshotError(f"feature snapshot must not contain outcome field: {key}")
            normalized[key] = _canonicalize(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise FeatureSnapshotError(f"unsupported snapshot value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if value.get(_SPECIAL_TYPE_KEY) == "decimal" and set(value) == {_SPECIAL_TYPE_KEY, "value"}:
            return Decimal(value["value"])
        if value.get(_SPECIAL_TYPE_KEY) == "datetime" and set(value) == {_SPECIAL_TYPE_KEY, "value"}:
            return _decode_datetime(value["value"], "snapshot datetime")
        return {str(key): _decode(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    canonical = _canonicalize(value)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _prepare(value: Any) -> Any:
    """Validate and copy semantic values while retaining supported Python types."""
    if value is None or isinstance(value, (str, bool, int, float, Decimal, datetime, Enum)):
        if isinstance(value, float) and not math.isfinite(value):
            raise FeatureSnapshotError("snapshot contains a non-finite float")
        if isinstance(value, Decimal) and not value.is_finite():
            raise FeatureSnapshotError("snapshot contains a non-finite Decimal")
        if isinstance(value, datetime):
            _require_aware_utc(value, "snapshot datetime")
        if isinstance(value, Enum):
            return value
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _prepare(asdict(value))
    module = type(value).__module__
    if module.startswith("numpy"):
        if hasattr(value, "tolist"):
            return _prepare(value.tolist())
        if hasattr(value, "item"):
            return _prepare(value.item())
    if isinstance(value, Mapping):
        prepared: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FeatureSnapshotError("snapshot mappings must use string keys")
            lowered = key.lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                raise FeatureSnapshotError(f"snapshot contains a prohibited sensitive field: {key}")
            if lowered in _OUTCOME_KEYS:
                raise FeatureSnapshotError(f"feature snapshot must not contain outcome field: {key}")
            prepared[key] = _prepare(item)
        return prepared
    if isinstance(value, (list, tuple)):
        return [_prepare(item) for item in value]
    raise FeatureSnapshotError(f"unsupported snapshot value type: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FeatureSnapshot:
    """Immutable decision-time feature state.

    The ``features`` mapping contains only information known at ``timestamp``.
    Outcome information belongs to LearningRecord lifecycle data and is rejected
    from this contract and from nested mappings.
    """

    timestamp: datetime
    symbol: str
    timeframe: str
    feature_engine_version: str
    feature_schema_version: str
    features: Mapping[str, Any]
    market_regime: str | None = None
    mtf_context: Mapping[str, Any] | None = None
    snapshot_hash: str = ""
    snapshot_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str | None = None
    provenance_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_aware_utc(self.timestamp, "timestamp")
        _require_aware_utc(self.created_at, "created_at")
        for field_name, value in (
            ("symbol", self.symbol),
            ("timeframe", self.timeframe),
            ("feature_engine_version", self.feature_engine_version),
            ("feature_schema_version", self.feature_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise FeatureSnapshotError(f"{field_name} must not be empty")
        if self.market_regime is not None and (not isinstance(self.market_regime, str) or not self.market_regime.strip()):
            raise FeatureSnapshotError("market_regime must be non-empty when provided")
        if self.dataset_version is not None and (not isinstance(self.dataset_version, str) or not self.dataset_version.strip()):
            raise FeatureSnapshotError("dataset_version must be non-empty when provided")
        if not isinstance(self.snapshot_id, str):
            raise FeatureSnapshotError("snapshot_id must be a string")
        if self.snapshot_id and not self.snapshot_id.strip():
            raise FeatureSnapshotError("snapshot_id must be non-empty when provided")

        canonical_features = _prepare(self.features)
        if not isinstance(canonical_features, dict):
            raise FeatureSnapshotError("features must be a mapping")
        canonical_mtf = None if self.mtf_context is None else _prepare(self.mtf_context)
        canonical_provenance = _prepare(self.provenance_metadata)
        if canonical_mtf is not None and not isinstance(canonical_mtf, dict):
            raise FeatureSnapshotError("mtf_context must be a mapping when provided")
        if not isinstance(canonical_provenance, dict):
            raise FeatureSnapshotError("provenance_metadata must be a mapping")

        object.__setattr__(self, "features", _freeze(canonical_features))
        object.__setattr__(self, "mtf_context", None if canonical_mtf is None else _freeze(canonical_mtf))
        object.__setattr__(self, "provenance_metadata", _freeze(canonical_provenance))

        expected_hash = self.calculate_hash()
        if self.snapshot_hash and self.snapshot_hash != expected_hash:
            raise FeatureSnapshotError("snapshot_hash does not match snapshot contents")
        object.__setattr__(self, "snapshot_hash", expected_hash)

        if self.snapshot_id:
            object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        else:
            deterministic_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"edge-hunter:feature-snapshot:{expected_hash}")
            )
            object.__setattr__(self, "snapshot_id", deterministic_id)

    @classmethod
    def from_market_analysis_snapshot(
        cls,
        snapshot: MarketAnalysisSnapshot,
        *,
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
        snapshot_id: str = "",
        created_at: datetime | None = None,
        dataset_version: str | None = None,
        provenance_metadata: Mapping[str, Any] | None = None,
    ) -> "FeatureSnapshot":
        """Adapt an existing Phase 03 snapshot without recalculating features."""
        if not isinstance(snapshot, MarketAnalysisSnapshot):
            raise FeatureSnapshotError("snapshot must be a MarketAnalysisSnapshot")
        metadata = snapshot.metadata or {}
        market_regime = metadata.get("market_regime")
        if market_regime is not None and not isinstance(market_regime, str):
            market_regime = None
        mtf_context = metadata.get("mtf_context")
        if mtf_context is not None and not isinstance(mtf_context, Mapping):
            raise FeatureSnapshotError("market analysis mtf_context must be a mapping")
        feature_engine_version = str(metadata.get("engine_version") or "unknown")
        return cls(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            feature_engine_version=feature_engine_version,
            feature_schema_version=feature_schema_version,
            features=snapshot.values,
            market_regime=market_regime,
            mtf_context=mtf_context,
            snapshot_id=snapshot_id,
            created_at=created_at or datetime.now(timezone.utc),
            dataset_version=dataset_version,
            provenance_metadata=provenance_metadata or {},
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Return only deterministic content that defines snapshot identity."""
        return {
            "timestamp": _utc_iso(self.timestamp),
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "feature_engine_version": self.feature_engine_version,
            "feature_schema_version": self.feature_schema_version,
            "features": _thaw(self.features),
            "market_regime": self.market_regime,
            "mtf_context": None if self.mtf_context is None else _thaw(self.mtf_context),
            "dataset_version": self.dataset_version,
        }

    def calculate_hash(self) -> str:
        """Calculate a stable SHA-256 over canonical decision-time content."""
        canonical = _canonical_json(self.canonical_payload()).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible semantic representation for callers."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": _utc_iso(self.timestamp),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_engine_version": self.feature_engine_version,
            "feature_schema_version": self.feature_schema_version,
            "features": _thaw(self.features),
            "market_regime": self.market_regime,
            "mtf_context": None if self.mtf_context is None else _thaw(self.mtf_context),
            "snapshot_hash": self.snapshot_hash,
            "created_at": _utc_iso(self.created_at),
            "dataset_version": self.dataset_version,
            "provenance_metadata": _thaw(self.provenance_metadata),
        }

    def to_storage_dict(self) -> dict[str, Any]:
        """Return a canonical form suitable for SQLite JSON columns."""
        payload = self.to_dict()
        return {
            **payload,
            "features": _canonicalize(payload["features"]),
            "mtf_context": None if payload["mtf_context"] is None else _canonicalize(payload["mtf_context"]),
            "provenance_metadata": _canonicalize(payload["provenance_metadata"]),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSnapshot":
        """Reconstruct a FeatureSnapshot from semantic or canonical JSON data."""
        if not isinstance(payload, Mapping):
            raise FeatureSnapshotError("snapshot payload must be a mapping")
        decoded = _decode(dict(payload))
        return cls(
            timestamp=_decode_datetime(str(decoded["timestamp"]), "timestamp"),
            symbol=str(decoded["symbol"]),
            timeframe=str(decoded["timeframe"]),
            feature_engine_version=str(decoded["feature_engine_version"]),
            feature_schema_version=str(decoded["feature_schema_version"]),
            features=decoded.get("features", {}),
            market_regime=decoded.get("market_regime"),
            mtf_context=decoded.get("mtf_context"),
            snapshot_hash=str(decoded.get("snapshot_hash", "")),
            snapshot_id=str(decoded.get("snapshot_id", "")),
            created_at=_decode_datetime(str(decoded["created_at"]), "created_at"),
            dataset_version=decoded.get("dataset_version"),
            provenance_metadata=decoded.get("provenance_metadata", {}),
        )

    def serialize(self) -> str:
        """Serialize this snapshot deterministically for storage/reload."""
        return _canonical_json(self.to_storage_dict())

    @classmethod
    def deserialize(cls, serialized: str) -> "FeatureSnapshot":
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise FeatureSnapshotError("invalid FeatureSnapshot JSON") from exc
        return cls.from_dict(payload)


class FeatureStore:
    """Small immutable SQLite Feature Store for decision-time snapshots."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, snapshot: FeatureSnapshot) -> FeatureSnapshot:
        """Persist a snapshot idempotently; never overwrite an immutable row."""
        existing_by_id = self.get(snapshot.snapshot_id)
        if existing_by_id is not None:
            if existing_by_id.snapshot_hash != snapshot.snapshot_hash:
                raise FeatureSnapshotImmutableError(
                    f"snapshot_id {snapshot.snapshot_id!r} already belongs to a different snapshot"
                )
            return existing_by_id

        existing_by_hash = self._get_by_hash(snapshot.snapshot_hash)
        if existing_by_hash is not None:
            return existing_by_hash

        payload = snapshot.to_dict()
        try:
            self.database.execute(
                """
                INSERT INTO feature_snapshots(
                    snapshot_id, timestamp, symbol, timeframe,
                    feature_engine_version, feature_schema_version,
                    features_json, market_regime, mtf_context_json,
                    snapshot_hash, created_at, dataset_version,
                    provenance_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    _utc_iso(snapshot.timestamp),
                    snapshot.symbol.upper(),
                    snapshot.timeframe,
                    snapshot.feature_engine_version,
                    snapshot.feature_schema_version,
                    _canonical_json(payload["features"]),
                    snapshot.market_regime,
                    None if payload["mtf_context"] is None else _canonical_json(payload["mtf_context"]),
                    snapshot.snapshot_hash,
                    _utc_iso(snapshot.created_at),
                    snapshot.dataset_version,
                    _canonical_json(payload["provenance_metadata"]),
                ),
            )
            self.database.commit()
        except sqlite3.IntegrityError as exc:
            # A concurrent writer can win the unique hash race; treat the exact
            # logical snapshot as an idempotent save, but surface other conflicts.
            existing = self._get_by_hash(snapshot.snapshot_hash)
            if existing is not None:
                return existing
            raise DuplicateFeatureSnapshotError("could not persist FeatureSnapshot") from exc
        return self.get(snapshot.snapshot_id)  # type: ignore[return-value]

    def get(self, snapshot_id: str) -> FeatureSnapshot | None:
        row = self.database.execute(
            "SELECT * FROM feature_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        return None if row is None else self._row_to_snapshot(row)

    def load(self, snapshot_id: str) -> FeatureSnapshot | None:
        """Alias for get(), kept explicit for reload workflows."""
        return self.get(snapshot_id)

    def exists(self, *, snapshot_id: str | None = None, snapshot_hash: str | None = None) -> bool:
        if snapshot_id is None and snapshot_hash is None:
            raise ValueError("snapshot_id or snapshot_hash is required")
        if snapshot_id is not None:
            row = self.database.execute(
                "SELECT 1 FROM feature_snapshots WHERE snapshot_id = ? LIMIT 1",
                (snapshot_id,),
            ).fetchone()
            if row is not None:
                return True
        if snapshot_hash is not None:
            row = self.database.execute(
                "SELECT 1 FROM feature_snapshots WHERE snapshot_hash = ? LIMIT 1",
                (snapshot_hash,),
            ).fetchone()
            return row is not None
        return False

    def find_by_identity(
        self,
        *,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        feature_engine_version: str,
        feature_schema_version: str,
    ) -> list[FeatureSnapshot]:
        _require_aware_utc(timestamp, "timestamp")
        rows = self.database.execute(
            """
            SELECT * FROM feature_snapshots
            WHERE symbol = ? AND timeframe = ? AND timestamp = ?
              AND feature_engine_version = ? AND feature_schema_version = ?
            ORDER BY snapshot_hash
            """,
            (
                symbol.upper(),
                timeframe,
                _utc_iso(timestamp),
                feature_engine_version,
                feature_schema_version,
            ),
        ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _get_by_hash(self, snapshot_hash: str) -> FeatureSnapshot | None:
        row = self.database.execute(
            "SELECT * FROM feature_snapshots WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        return None if row is None else self._row_to_snapshot(row)

    @staticmethod
    def _row_to_snapshot(row: Any) -> FeatureSnapshot:
        mtf_payload = None if row["mtf_context_json"] is None else json.loads(row["mtf_context_json"])
        return FeatureSnapshot(
            snapshot_id=row["snapshot_id"],
            timestamp=_decode_datetime(row["timestamp"], "timestamp"),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            feature_engine_version=row["feature_engine_version"],
            feature_schema_version=row["feature_schema_version"],
            features=json.loads(row["features_json"]),
            market_regime=row["market_regime"],
            mtf_context=mtf_payload,
            snapshot_hash=row["snapshot_hash"],
            created_at=_decode_datetime(row["created_at"], "created_at"),
            dataset_version=row["dataset_version"],
            provenance_metadata=json.loads(row["provenance_metadata_json"]),
        )


__all__ = [
    "DuplicateFeatureSnapshotError",
    "FeatureSnapshot",
    "FeatureSnapshotError",
    "FeatureSnapshotImmutableError",
    "FeatureStore",
    "FEATURE_SCHEMA_VERSION",
]

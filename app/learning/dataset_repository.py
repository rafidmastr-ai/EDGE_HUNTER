"""Immutable SQLite persistence for EDGE HUNTER L5 dataset versions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.db.database import Database
from app.learning.dataset import DatasetRow, DatasetSplit, DatasetVersion
from app.learning.feature_store import FeatureStore
from app.learning.models import LearningSourceType


class DatasetVersionPersistenceError(ValueError):
    """Raised when a dataset version cannot be persisted or reloaded safely."""


class DatasetVersionImmutableError(DatasetVersionPersistenceError):
    """Raised when an existing immutable dataset version conflicts with a new one."""


class DatasetVersionRepository:
    """Create/read-only persistence for immutable DatasetVersion manifests and row references."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, version: DatasetVersion, rows: tuple[DatasetRow, ...] | list[DatasetRow]) -> DatasetVersion:
        existing = self.get(version.dataset_version_id)
        if existing is not None:
            if existing.fingerprint != version.fingerprint:
                raise DatasetVersionImmutableError(
                    f"dataset_version_id {version.dataset_version_id!r} already belongs to another fingerprint"
                )
            self._validate_existing_rows(version.dataset_version_id, rows)
            return existing

        row_list = tuple(rows)
        if len(row_list) != version.row_count:
            raise DatasetVersionPersistenceError("row count does not match DatasetVersion")
        if len({row.record_id for row in row_list}) != len(row_list):
            raise DatasetVersionPersistenceError("duplicate record_id in DatasetVersion rows")

        try:
            self.database.execute(
                """
                INSERT INTO learning_dataset_versions(
                    dataset_version_id, dataset_name, created_at,
                    feature_schema_version, label_version, dataset_policy_version,
                    split_policy_version, row_count, train_count, validation_count,
                    oos_count, start_timestamp, end_timestamp, fingerprint,
                    total_records, included_records, excluded_records,
                    statistics_json, filter_scope_json, split_policy_json,
                    exclusion_counts_json, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.dataset_version_id,
                    version.dataset_name,
                    version.created_at.astimezone(timezone.utc).isoformat(),
                    version.feature_schema_version,
                    version.label_version,
                    version.dataset_policy_version,
                    version.split_policy_version,
                    version.row_count,
                    version.train_count,
                    version.validation_count,
                    version.oos_count,
                    None if version.start_timestamp is None else version.start_timestamp.astimezone(timezone.utc).isoformat(),
                    None if version.end_timestamp is None else version.end_timestamp.astimezone(timezone.utc).isoformat(),
                    version.fingerprint,
                    version.total_records,
                    version.included_records,
                    version.excluded_records,
                    _json(version.statistics),
                    _json(version.filter_scope),
                    _json(version.split_policy),
                    _json(version.exclusion_counts),
                    _json(version.manifest()),
                ),
            )
            for row in row_list:
                self.database.execute(
                    """
                    INSERT INTO learning_dataset_rows(
                        dataset_version_id, record_id, split, decision_timestamp,
                        symbol, timeframe, strategy_name, strategy_variant,
                        strategy_version, feature_snapshot_id, feature_snapshot_hash,
                        feature_schema_version, outcome, label_version, source_type,
                        row_fingerprint, row_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version.dataset_version_id,
                        row.record_id,
                        row.split.value,
                        row.decision_timestamp.astimezone(timezone.utc).isoformat(),
                        row.symbol.upper(),
                        row.timeframe,
                        row.strategy_name,
                        row.strategy_variant,
                        row.strategy_version,
                        row.feature_snapshot_id,
                        row.feature_snapshot_hash,
                        row.feature_schema_version,
                        row.outcome,
                        row.label_version,
                        row.source_type.value,
                        row.row_fingerprint,
                        _json({
                            "direction": row.direction,
                            "entry_price": row.entry_price,
                            "target": row.target,
                            "stop_loss": row.stop_loss,
                            "risk_reward": row.risk_reward,
                            "confidence": row.confidence,
                            "parameters_snapshot": row.parameters_snapshot,
                            "evidence_snapshot": row.evidence_snapshot,
                            "market_regime": row.market_regime,
                            "exit_timestamp": None if row.exit_timestamp is None else row.exit_timestamp.isoformat(),
                            "exit_price": row.exit_price,
                            "exit_reason": row.exit_reason,
                            "duration_seconds": row.duration_seconds,
                            "provenance_metadata": row.provenance_metadata,
                        }),
                    ),
                )
            self.database.commit()
        except sqlite3.IntegrityError as exc:
            self.database.connection.rollback()
            existing = self.get(version.dataset_version_id)
            if existing is not None and existing.fingerprint == version.fingerprint:
                return existing
            raise DatasetVersionPersistenceError("could not persist dataset version") from exc
        return self.get(version.dataset_version_id) or version

    def get(self, dataset_version_id: str) -> DatasetVersion | None:
        row = self.database.execute(
            "SELECT * FROM learning_dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchone()
        return None if row is None else self._version_from_row(row)

    def list(self, *, dataset_name: str | None = None) -> list[DatasetVersion]:
        if dataset_name is None:
            rows = self.database.execute(
                "SELECT * FROM learning_dataset_versions ORDER BY created_at ASC, dataset_version_id ASC"
            ).fetchall()
        else:
            rows = self.database.execute(
                "SELECT * FROM learning_dataset_versions WHERE dataset_name = ? ORDER BY created_at ASC, dataset_version_id ASC",
                (dataset_name,),
            ).fetchall()
        return [self._version_from_row(row) for row in rows]

    def list_rows(
        self,
        dataset_version_id: str,
        *,
        feature_store: FeatureStore | None = None,
    ) -> tuple[DatasetRow, ...]:
        version = self.get(dataset_version_id)
        if version is None:
            raise KeyError(dataset_version_id)
        rows = self.database.execute(
            "SELECT * FROM learning_dataset_rows WHERE dataset_version_id = ? ORDER BY decision_timestamp ASC, record_id ASC",
            (dataset_version_id,),
        ).fetchall()
        if feature_store is None:
            raise DatasetVersionPersistenceError("feature_store is required to reload DatasetRow features")
        result: list[DatasetRow] = []
        for row in rows:
            snapshot = feature_store.get(row["feature_snapshot_id"])
            if snapshot is None:
                raise DatasetVersionPersistenceError(
                    f"feature snapshot missing while reloading dataset row {row['record_id']}"
                )
            metadata = json.loads(row["row_metadata_json"])
            payload = {
                "record_id": row["record_id"],
                "split": DatasetSplit(row["split"]),
                "decision_timestamp": _dt(row["decision_timestamp"]),
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "strategy_name": row["strategy_name"],
                "strategy_variant": row["strategy_variant"],
                "strategy_version": row["strategy_version"],
                "direction": metadata["direction"],
                "entry_price": metadata["entry_price"],
                "target": metadata["target"],
                "stop_loss": metadata["stop_loss"],
                "risk_reward": metadata["risk_reward"],
                "confidence": metadata["confidence"],
                "parameters_snapshot": metadata["parameters_snapshot"],
                "evidence_snapshot": metadata["evidence_snapshot"],
                "market_regime": metadata["market_regime"],
                "feature_snapshot_id": snapshot.snapshot_id,
                "feature_snapshot_hash": row["feature_snapshot_hash"],
                "feature_schema_version": row["feature_schema_version"],
                "features": snapshot.features,
                "outcome": row["outcome"],
                "label_version": row["label_version"],
                "exit_timestamp": None if metadata["exit_timestamp"] is None else _dt(metadata["exit_timestamp"]),
                "exit_price": metadata["exit_price"],
                "exit_reason": metadata["exit_reason"],
                "duration_seconds": metadata["duration_seconds"],
                "source_type": LearningSourceType(row["source_type"]),
                "provenance_metadata": metadata["provenance_metadata"],
            }
            reconstructed = DatasetRow(**payload)
            if reconstructed.row_fingerprint != row["row_fingerprint"]:
                raise DatasetVersionPersistenceError(f"row fingerprint mismatch for {row['record_id']}")
            result.append(reconstructed)
        if len(result) != version.row_count:
            raise DatasetVersionPersistenceError("persisted row count does not match DatasetVersion")
        return tuple(result)

    def _validate_existing_rows(self, dataset_version_id: str, rows: tuple[DatasetRow, ...] | list[DatasetRow]) -> None:
        expected = {row.record_id: row.row_fingerprint for row in rows}
        current = self.database.execute(
            "SELECT record_id, row_fingerprint FROM learning_dataset_rows WHERE dataset_version_id = ?",
            (dataset_version_id,),
        ).fetchall()
        actual = {row["record_id"]: row["row_fingerprint"] for row in current}
        if actual != expected:
            raise DatasetVersionImmutableError("existing DatasetVersion rows do not match the requested immutable version")

    @staticmethod
    def _version_from_row(row: Any) -> DatasetVersion:
        return DatasetVersion(
            dataset_version_id=row["dataset_version_id"],
            dataset_name=row["dataset_name"],
            created_at=_dt(row["created_at"]),
            feature_schema_version=row["feature_schema_version"],
            label_version=row["label_version"],
            dataset_policy_version=row["dataset_policy_version"],
            split_policy_version=row["split_policy_version"],
            row_count=int(row["row_count"]),
            train_count=int(row["train_count"]),
            validation_count=int(row["validation_count"]),
            oos_count=int(row["oos_count"]),
            start_timestamp=None if row["start_timestamp"] is None else _dt(row["start_timestamp"]),
            end_timestamp=None if row["end_timestamp"] is None else _dt(row["end_timestamp"]),
            fingerprint=row["fingerprint"],
            total_records=int(row["total_records"]),
            included_records=int(row["included_records"]),
            excluded_records=int(row["excluded_records"]),
            statistics=json.loads(row["statistics_json"]),
            filter_scope=json.loads(row["filter_scope_json"]),
            split_policy=json.loads(row["split_policy_json"]),
            exclusion_counts=json.loads(row["exclusion_counts_json"]),
        )


def _dt(value: str | None) -> datetime:
    if value is None:
        raise DatasetVersionPersistenceError("dataset timestamp must not be null")
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "DatasetVersionImmutableError",
    "DatasetVersionPersistenceError",
    "DatasetVersionRepository",
]

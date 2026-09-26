"""SQLite persistence for immutable L9 LearnedPolicyVersion records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.database import Database
from app.learning.policies import LearnedPolicyError, LearnedPolicyVersion, PolicyMode, PolicyType


class LearnedPolicyPersistenceError(LearnedPolicyError):
    """Raised when policy persistence violates an immutable contract."""


class LearnedPolicyRepository:
    """Insert/read-only persistence for LearnedPolicyVersion."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, policy: LearnedPolicyVersion) -> LearnedPolicyVersion:
        existing = self.get(policy.policy_version_id)
        if existing is not None:
            if existing.to_dict() != policy.to_dict():
                raise LearnedPolicyPersistenceError("policy_version_id is already associated with another immutable payload")
            return existing
        self.database.execute(
            """
            INSERT INTO learning_policy_versions(
                policy_version_id, policy_version_label, strategy_name, strategy_variant,
                strategy_version, policy_types_json, source_model_version_id,
                source_training_run_id, source_dataset_version_id,
                feature_schema_version, label_version, policy_parameters_json,
                enabled_capabilities_json, fingerprint, created_at, metadata_json,
                default_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.policy_version_id,
                policy.policy_version_label,
                policy.strategy_name,
                policy.strategy_variant,
                policy.strategy_version,
                _json([item.value for item in policy.policy_types]),
                policy.source_model_version_id,
                policy.source_training_run_id,
                policy.source_dataset_version_id,
                policy.feature_schema_version,
                policy.label_version,
                _json(policy.policy_parameters),
                _json(policy.enabled_capabilities),
                policy.fingerprint,
                _iso(policy.created_at),
                _json(policy.metadata),
                policy.default_mode.value,
            ),
        )
        self.database.commit()
        return self.get(policy.policy_version_id) or policy

    def get(self, policy_version_id: str) -> LearnedPolicyVersion | None:
        row = self.database.execute(
            "SELECT * FROM learning_policy_versions WHERE policy_version_id = ?",
            (policy_version_id,),
        ).fetchone()
        return None if row is None else _from_row(row)

    def list(self, *, strategy_name: str | None = None) -> list[LearnedPolicyVersion]:
        if strategy_name is None:
            rows = self.database.execute(
                "SELECT * FROM learning_policy_versions ORDER BY created_at ASC, policy_version_id ASC"
            ).fetchall()
        else:
            rows = self.database.execute(
                "SELECT * FROM learning_policy_versions WHERE strategy_name = ? ORDER BY created_at ASC, policy_version_id ASC",
                (strategy_name,),
            ).fetchall()
        return [_from_row(row) for row in rows]

    def find_by_strategy(self, strategy_name: str) -> list[LearnedPolicyVersion]:
        return self.list(strategy_name=strategy_name)

    def find_by_model(self, model_version_id: str) -> list[LearnedPolicyVersion]:
        rows = self.database.execute(
            "SELECT * FROM learning_policy_versions WHERE source_model_version_id = ? ORDER BY created_at ASC, policy_version_id ASC",
            (model_version_id,),
        ).fetchall()
        return [_from_row(row) for row in rows]

    def find_by_dataset(self, dataset_version_id: str) -> list[LearnedPolicyVersion]:
        rows = self.database.execute(
            "SELECT * FROM learning_policy_versions WHERE source_dataset_version_id = ? ORDER BY created_at ASC, policy_version_id ASC",
            (dataset_version_id,),
        ).fetchall()
        return [_from_row(row) for row in rows]


def _from_row(row: Any) -> LearnedPolicyVersion:
    return LearnedPolicyVersion(
        policy_version_id=row["policy_version_id"],
        policy_version_label=row["policy_version_label"],
        strategy_name=row["strategy_name"],
        strategy_variant=row["strategy_variant"],
        strategy_version=row["strategy_version"],
        policy_types=tuple(PolicyType(item) for item in json.loads(row["policy_types_json"])),
        source_model_version_id=row["source_model_version_id"],
        source_training_run_id=row["source_training_run_id"],
        source_dataset_version_id=row["source_dataset_version_id"],
        feature_schema_version=row["feature_schema_version"],
        label_version=row["label_version"],
        policy_parameters=json.loads(row["policy_parameters_json"]),
        enabled_capabilities=tuple(str(item) for item in json.loads(row["enabled_capabilities_json"])),
        fingerprint=row["fingerprint"],
        created_at=_dt(row["created_at"]),
        metadata=json.loads(row["metadata_json"]),
        default_mode=PolicyMode(row["default_mode"]),
    )


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["LearnedPolicyPersistenceError", "LearnedPolicyRepository"]

"""SQLite persistence for L6 TrainingRuns; no model registry semantics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.database import Database
from app.learning.training import TrainingPipelineError, TrainingRun, TrainingStatus


class TrainingRunPersistenceError(TrainingPipelineError):
    """Raised when TrainingRun persistence is inconsistent or unsafe."""


class TrainingRunRepository:
    """Create/read/update-status persistence for offline TrainingRun records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, run: TrainingRun) -> TrainingRun:
        existing = self.get(run.training_run_id)
        if existing is not None:
            if existing.training_identity_hash != run.training_identity_hash:
                raise TrainingRunPersistenceError("training_run_id is already associated with another identity")
            return existing
        self.database.execute(
            """
            INSERT INTO learning_training_runs(
                training_run_id, training_identity_hash, dataset_version_id,
                started_at, completed_at, status,
                strategy_name, strategy_variant, strategy_version,
                model_type, model_version_label,
                feature_schema_version, label_version,
                train_count, validation_count, oos_count,
                training_config_json, random_seed, metrics_json,
                artifact_path, artifact_sha256, artifact_metadata_json,
                error_type, error_message, error_stage, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.training_run_id,
                run.training_identity_hash,
                run.dataset_version_id,
                _iso(run.started_at),
                _iso(run.completed_at),
                run.status.value,
                run.strategy_name,
                run.strategy_variant,
                run.strategy_version,
                run.model_type,
                run.model_version_label,
                run.feature_schema_version,
                run.label_version,
                run.train_count,
                run.validation_count,
                run.oos_count,
                _json(run.training_config),
                run.random_seed,
                _json(run.metrics),
                run.artifact_path,
                run.artifact_sha256,
                _json(run.artifact_metadata),
                run.error_type,
                run.error_message,
                run.error_stage,
                _iso(run.created_at),
            ),
        )
        self.database.commit()
        return self.get(run.training_run_id) or run

    def mark_running(self, training_run_id: str, *, started_at: datetime) -> TrainingRun:
        current = self.get(training_run_id)
        if current is None:
            raise KeyError(training_run_id)
        if current.status not in {TrainingStatus.PENDING, TrainingStatus.RUNNING}:
            raise TrainingRunPersistenceError(f"cannot mark {current.status.value} run as RUNNING")
        self.database.execute(
            "UPDATE learning_training_runs SET status = 'RUNNING', started_at = ? WHERE training_run_id = ?",
            (_iso(started_at), training_run_id),
        )
        self.database.commit()
        return self.get(training_run_id)  # type: ignore[return-value]

    def mark_completed(self, run: TrainingRun) -> TrainingRun:
        if run.status != TrainingStatus.COMPLETED:
            raise TrainingRunPersistenceError("mark_completed requires COMPLETED status")
        current = self.get(run.training_run_id)
        if current is None:
            raise KeyError(run.training_run_id)
        if current.status == TrainingStatus.COMPLETED:
            if current.to_dict() != run.to_dict():
                raise TrainingRunPersistenceError("COMPLETED TrainingRun cannot be overwritten")
            return current
        if current.status not in {TrainingStatus.RUNNING, TrainingStatus.PENDING}:
            raise TrainingRunPersistenceError(f"cannot complete {current.status.value} run")
        self.database.execute(
            """
            UPDATE learning_training_runs
            SET status = 'COMPLETED', completed_at = ?, train_count = ?,
                validation_count = ?, oos_count = ?, metrics_json = ?,
                artifact_path = ?, artifact_sha256 = ?, artifact_metadata_json = ?,
                error_type = NULL, error_message = NULL, error_stage = NULL
            WHERE training_run_id = ? AND status IN ('PENDING', 'RUNNING')
            """,
            (
                _iso(run.completed_at),
                run.train_count,
                run.validation_count,
                run.oos_count,
                _json(run.metrics),
                run.artifact_path,
                run.artifact_sha256,
                _json(run.artifact_metadata),
                run.training_run_id,
            ),
        )
        self.database.commit()
        return self.get(run.training_run_id)  # type: ignore[return-value]

    def mark_failed(
        self,
        training_run_id: str,
        *,
        completed_at: datetime,
        error_type: str,
        error_message: str,
        error_stage: str,
    ) -> TrainingRun:
        current = self.get(training_run_id)
        if current is None:
            raise KeyError(training_run_id)
        if current.status == TrainingStatus.COMPLETED:
            return current
        self.database.execute(
            """
            UPDATE learning_training_runs
            SET status = 'FAILED', completed_at = ?,
                error_type = ?, error_message = ?, error_stage = ?
            WHERE training_run_id = ? AND status IN ('PENDING', 'RUNNING')
            """,
            (_iso(completed_at), error_type, error_message, error_stage, training_run_id),
        )
        self.database.commit()
        return self.get(training_run_id)  # type: ignore[return-value]

    def get(self, training_run_id: str) -> TrainingRun | None:
        row = self.database.execute(
            "SELECT * FROM learning_training_runs WHERE training_run_id = ?",
            (training_run_id,),
        ).fetchone()
        return None if row is None else _from_row(row)

    def list(self, *, dataset_version_id: str | None = None) -> list[TrainingRun]:
        if dataset_version_id is None:
            rows = self.database.execute(
                "SELECT * FROM learning_training_runs ORDER BY created_at ASC, training_run_id ASC"
            ).fetchall()
        else:
            rows = self.database.execute(
                "SELECT * FROM learning_training_runs WHERE dataset_version_id = ? ORDER BY created_at ASC, training_run_id ASC",
                (dataset_version_id,),
            ).fetchall()
        return [_from_row(row) for row in rows]


def _from_row(row: Any) -> TrainingRun:
    return TrainingRun(
        training_run_id=row["training_run_id"],
        training_identity_hash=row["training_identity_hash"],
        dataset_version_id=row["dataset_version_id"],
        started_at=_dt(row["started_at"]),
        completed_at=None if row["completed_at"] is None else _dt(row["completed_at"]),
        status=TrainingStatus(row["status"]),
        strategy_name=row["strategy_name"],
        strategy_variant=row["strategy_variant"],
        strategy_version=row["strategy_version"],
        model_type=row["model_type"],
        model_version_label=row["model_version_label"],
        feature_schema_version=row["feature_schema_version"],
        label_version=row["label_version"],
        train_count=int(row["train_count"]),
        validation_count=int(row["validation_count"]),
        oos_count=int(row["oos_count"]),
        training_config=json.loads(row["training_config_json"]),
        random_seed=int(row["random_seed"]),
        metrics=json.loads(row["metrics_json"]),
        artifact_path=row["artifact_path"],
        artifact_sha256=row["artifact_sha256"],
        artifact_metadata=json.loads(row["artifact_metadata_json"]),
        error_type=row["error_type"],
        error_message=row["error_message"],
        error_stage=row["error_stage"],
        created_at=_dt(row["created_at"]),
    )


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["TrainingRunPersistenceError", "TrainingRunRepository"]

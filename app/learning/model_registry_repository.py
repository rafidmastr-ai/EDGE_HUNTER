"""SQLite persistence for L7 ModelVersions and Evaluations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.database import Database
from app.learning.registry import EvaluationStatus, ModelEvaluation, ModelRegistryError, ModelVersion, ModelVersionStatus


class ModelRegistryPersistenceError(ModelRegistryError):
    pass


class ModelRegistryRepository:
    """Insert/read ModelVersions and immutable Evaluation records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, model: ModelVersion) -> ModelVersion:
        existing = self.get(model.model_version_id)
        if existing is not None:
            if _immutable_payload(existing) != _immutable_payload(model):
                raise ModelRegistryPersistenceError("ModelVersion identity already belongs to another immutable payload")
            return existing
        try:
            self.database.execute(
                """
                INSERT INTO learning_model_versions(
                    model_version_id, model_version_label, strategy_name, strategy_variant,
                    strategy_version, model_type, training_run_id, dataset_version_id,
                    feature_schema_version, label_version, artifact_path, artifact_sha256,
                    artifact_format_version, created_at, registered_at, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.model_version_id, model.model_version_label, model.strategy_name, model.strategy_variant,
                    model.strategy_version, model.model_type, model.training_run_id, model.dataset_version_id,
                    model.feature_schema_version, model.label_version, model.artifact_path, model.artifact_sha256,
                    model.artifact_format_version, _iso(model.created_at), _iso(model.registered_at),
                    model.status.value, _json(model.metadata),
                ),
            )
            self.database.commit()
        except Exception as exc:
            self.database.connection.rollback()
            raise ModelRegistryPersistenceError("could not persist ModelVersion") from exc
        return self.get(model.model_version_id) or model

    def get(self, model_version_id: str) -> ModelVersion | None:
        row = self.database.execute("SELECT * FROM learning_model_versions WHERE model_version_id = ?", (model_version_id,)).fetchone()
        return None if row is None else _model_from_row(row)

    def list(
        self,
        *,
        strategy_name: str | None = None,
        dataset_version_id: str | None = None,
        status: ModelVersionStatus | str | None = None,
    ) -> list[ModelVersion]:
        clauses: list[str] = []
        params: list[Any] = []
        if strategy_name is not None:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if dataset_version_id is not None:
            clauses.append("dataset_version_id = ?")
            params.append(dataset_version_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value if isinstance(status, ModelVersionStatus) else str(status))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.database.execute(
            f"SELECT * FROM learning_model_versions{where} ORDER BY registered_at ASC, model_version_id ASC", tuple(params)
        ).fetchall()
        return [_model_from_row(row) for row in rows]

    def find_by_strategy(self, strategy_name: str) -> list[ModelVersion]:
        return self.list(strategy_name=strategy_name)

    def find_by_training_run(self, training_run_id: str) -> list[ModelVersion]:
        rows = self.database.execute(
            "SELECT * FROM learning_model_versions WHERE training_run_id = ? ORDER BY registered_at ASC, model_version_id ASC",
            (training_run_id,),
        ).fetchall()
        return [_model_from_row(row) for row in rows]

    def find_by_dataset(self, dataset_version_id: str) -> list[ModelVersion]:
        return self.list(dataset_version_id=dataset_version_id)

    def find_by_status(self, status: ModelVersionStatus | str) -> list[ModelVersion]:
        return self.list(status=status)

    def mark_evaluated(self, model_version_id: str) -> ModelVersion:
        current = self.get(model_version_id)
        if current is None:
            raise KeyError(model_version_id)
        if current.status == ModelVersionStatus.EVALUATED:
            return current
        if current.status == ModelVersionStatus.REVOKED:
            raise ModelRegistryPersistenceError("revoked ModelVersion cannot be evaluated")
        self.database.execute(
            "UPDATE learning_model_versions SET status = 'EVALUATED' WHERE model_version_id = ?",
            (model_version_id,),
        )
        self.database.commit()
        return self.get(model_version_id)  # type: ignore[return-value]

    def revoke(self, model_version_id: str) -> ModelVersion:
        current = self.get(model_version_id)
        if current is None:
            raise KeyError(model_version_id)
        if current.status == ModelVersionStatus.REVOKED:
            return current
        self.database.execute(
            "UPDATE learning_model_versions SET status = 'REVOKED' WHERE model_version_id = ?",
            (model_version_id,),
        )
        self.database.commit()
        return self.get(model_version_id)  # type: ignore[return-value]

    def create_evaluation(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        existing = self.get_evaluation(evaluation.evaluation_id)
        if existing is not None:
            if existing.evaluation_fingerprint != evaluation.evaluation_fingerprint:
                raise ModelRegistryPersistenceError("evaluation identity conflict")
            return existing
        self.database.execute(
            """
            INSERT INTO learning_model_evaluations(
                evaluation_id, model_version_id, dataset_version_id, split, evaluated_at,
                sample_count, status, metrics_json, confusion_matrix_json, class_distribution_json,
                feature_schema_version, label_version, evaluation_config_json,
                evaluation_fingerprint, started_at, completed_at, error_type, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation.evaluation_id, evaluation.model_version_id, evaluation.dataset_version_id,
                evaluation.split, _iso(evaluation.evaluated_at), evaluation.sample_count,
                evaluation.status.value, _json(evaluation.metrics), _json(evaluation.confusion_matrix),
                _json(evaluation.class_distribution), evaluation.feature_schema_version, evaluation.label_version,
                _json(evaluation.evaluation_config), evaluation.evaluation_fingerprint,
                _iso(evaluation.started_at), _iso(evaluation.completed_at), evaluation.error_type, evaluation.error_message,
            ),
        )
        self.database.commit()
        return self.get_evaluation(evaluation.evaluation_id) or evaluation

    def get_evaluation(self, evaluation_id: str) -> ModelEvaluation | None:
        row = self.database.execute("SELECT * FROM learning_model_evaluations WHERE evaluation_id = ?", (evaluation_id,)).fetchone()
        return None if row is None else _evaluation_from_row(row)

    def get_evaluations(self, model_version_id: str) -> list[ModelEvaluation]:
        rows = self.database.execute(
            "SELECT * FROM learning_model_evaluations WHERE model_version_id = ? ORDER BY evaluated_at ASC, evaluation_id ASC",
            (model_version_id,),
        ).fetchall()
        return [_evaluation_from_row(row) for row in rows]

    def mark_evaluation_completed(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        self._replace_evaluation(evaluation, expected_status=EvaluationStatus.RUNNING)
        return self.get_evaluation(evaluation.evaluation_id)  # type: ignore[return-value]

    def mark_evaluation_failed(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        self._replace_evaluation(evaluation, expected_status=EvaluationStatus.RUNNING)
        return self.get_evaluation(evaluation.evaluation_id)  # type: ignore[return-value]

    def _replace_evaluation(self, evaluation: ModelEvaluation, *, expected_status: EvaluationStatus) -> None:
        current = self.get_evaluation(evaluation.evaluation_id)
        if current is None:
            raise KeyError(evaluation.evaluation_id)
        if current.status != expected_status:
            if current.status == evaluation.status and current.to_dict() == evaluation.to_dict():
                return
            raise ModelRegistryPersistenceError("evaluation cannot be overwritten from its current status")
        self.database.execute(
            """
            UPDATE learning_model_evaluations
            SET evaluated_at = ?, sample_count = ?, status = ?, metrics_json = ?,
                confusion_matrix_json = ?, class_distribution_json = ?, evaluation_config_json = ?,
                completed_at = ?, error_type = ?, error_message = ?
            WHERE evaluation_id = ? AND status = ?
            """,
            (
                _iso(evaluation.evaluated_at), evaluation.sample_count, evaluation.status.value,
                _json(evaluation.metrics), _json(evaluation.confusion_matrix), _json(evaluation.class_distribution),
                _json(evaluation.evaluation_config), _iso(evaluation.completed_at), evaluation.error_type,
                evaluation.error_message, evaluation.evaluation_id, expected_status.value,
            ),
        )
        self.database.commit()


def _immutable_payload(model: ModelVersion) -> tuple[Any, ...]:
    return (
        model.model_version_label, model.strategy_name, model.strategy_variant, model.strategy_version,
        model.model_type, model.training_run_id, model.dataset_version_id, model.feature_schema_version,
        model.label_version, model.artifact_path, model.artifact_sha256, model.artifact_format_version,
        _json(model.metadata),
    )


def _model_from_row(row: Any) -> ModelVersion:
    return ModelVersion(
        model_version_id=row["model_version_id"],
        model_version_label=row["model_version_label"],
        strategy_name=row["strategy_name"],
        strategy_variant=row["strategy_variant"],
        strategy_version=row["strategy_version"],
        model_type=row["model_type"],
        training_run_id=row["training_run_id"],
        dataset_version_id=row["dataset_version_id"],
        feature_schema_version=row["feature_schema_version"],
        label_version=row["label_version"],
        artifact_path=row["artifact_path"],
        artifact_sha256=row["artifact_sha256"],
        artifact_format_version=row["artifact_format_version"],
        created_at=_dt(row["created_at"]),
        registered_at=_dt(row["registered_at"]),
        status=ModelVersionStatus(row["status"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _evaluation_from_row(row: Any) -> ModelEvaluation:
    return ModelEvaluation(
        evaluation_id=row["evaluation_id"],
        model_version_id=row["model_version_id"],
        dataset_version_id=row["dataset_version_id"],
        split=row["split"],
        evaluated_at=_dt(row["evaluated_at"]),
        sample_count=int(row["sample_count"]),
        status=EvaluationStatus(row["status"]),
        metrics=json.loads(row["metrics_json"]),
        confusion_matrix=tuple(tuple(int(item) for item in values) for values in json.loads(row["confusion_matrix_json"])),
        class_distribution={str(key): int(value) for key, value in json.loads(row["class_distribution_json"]).items()},
        feature_schema_version=row["feature_schema_version"],
        label_version=row["label_version"],
        evaluation_config=json.loads(row["evaluation_config_json"]),
        evaluation_fingerprint=row["evaluation_fingerprint"],
        started_at=_dt(row["started_at"]),
        completed_at=None if row["completed_at"] is None else _dt(row["completed_at"]),
        error_type=row["error_type"],
        error_message=row["error_message"],
    )


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["ModelRegistryPersistenceError", "ModelRegistryRepository"]

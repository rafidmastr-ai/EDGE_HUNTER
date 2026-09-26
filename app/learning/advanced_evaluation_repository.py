"""SQLite persistence for immutable L8 evaluation reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.database import Database
from app.learning.oos_evaluation import (
    L8EvaluationReport,
    L8EvaluationStatus,
    L8EvaluationType,
)


class L8EvaluationPersistenceError(ValueError):
    """Raised when an L8 evaluation report cannot be persisted safely."""


class L8EvaluationRepository:
    """Create/read repository with deterministic fingerprint idempotency."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, report: L8EvaluationReport) -> L8EvaluationReport:
        existing = self.get(report.evaluation_id)
        if existing is not None:
            if existing.evaluation_fingerprint != report.evaluation_fingerprint:
                raise L8EvaluationPersistenceError("evaluation_id is already associated with another fingerprint")
            return existing
        self.database.execute(
            """
            INSERT INTO learning_l8_evaluations(
                evaluation_id, evaluation_type, model_version_id, dataset_version_id,
                split, evaluated_at, status, sample_count, eligible_samples,
                excluded_samples, metrics_json, confusion_matrix_json,
                class_distribution_json, feature_schema_version, label_version,
                evaluation_config_json, evaluation_fingerprint, metadata_json,
                windows_json, stability_summary_json, started_at, completed_at,
                error_type, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.evaluation_id,
                report.evaluation_type.value,
                report.model_version_id,
                report.dataset_version_id,
                report.split,
                _iso(report.evaluated_at),
                report.status.value,
                report.sample_count,
                report.eligible_samples,
                report.excluded_samples,
                _json(report.metrics),
                _json(report.confusion_matrix),
                _json(report.class_distribution),
                report.feature_schema_version,
                report.label_version,
                _json(report.evaluation_config),
                report.evaluation_fingerprint,
                _json(report.metadata),
                _json(report.windows),
                _json(report.stability_summary),
                _iso(report.started_at),
                _iso(report.completed_at),
                report.error_type,
                report.error_message,
                _iso(datetime.now(timezone.utc)),
            ),
        )
        self.database.commit()
        return self.get(report.evaluation_id) or report

    def get(self, evaluation_id: str) -> L8EvaluationReport | None:
        row = self.database.execute(
            "SELECT * FROM learning_l8_evaluations WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
        return None if row is None else _from_row(row)

    def list(
        self,
        *,
        model_version_id: str | None = None,
        dataset_version_id: str | None = None,
        evaluation_type: L8EvaluationType | str | None = None,
        status: L8EvaluationStatus | str | None = None,
    ) -> list[L8EvaluationReport]:
        clauses: list[str] = []
        params: list[Any] = []
        if model_version_id is not None:
            clauses.append("model_version_id = ?")
            params.append(model_version_id)
        if dataset_version_id is not None:
            clauses.append("dataset_version_id = ?")
            params.append(dataset_version_id)
        if evaluation_type is not None:
            clauses.append("evaluation_type = ?")
            params.append(evaluation_type.value if isinstance(evaluation_type, L8EvaluationType) else str(evaluation_type))
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value if isinstance(status, L8EvaluationStatus) else str(status))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.database.execute(
            f"SELECT * FROM learning_l8_evaluations{where} ORDER BY evaluated_at ASC, evaluation_id ASC",
            tuple(params),
        ).fetchall()
        return [_from_row(row) for row in rows]

    def find_by_model(self, model_version_id: str) -> list[L8EvaluationReport]:
        return self.list(model_version_id=model_version_id)

    def find_by_dataset(self, dataset_version_id: str) -> list[L8EvaluationReport]:
        return self.list(dataset_version_id=dataset_version_id)

    def find_by_type(self, evaluation_type: L8EvaluationType | str) -> list[L8EvaluationReport]:
        return self.list(evaluation_type=evaluation_type)


def _from_row(row: Any) -> L8EvaluationReport:
    return L8EvaluationReport(
        evaluation_id=row["evaluation_id"],
        evaluation_type=L8EvaluationType(row["evaluation_type"]),
        model_version_id=row["model_version_id"],
        dataset_version_id=row["dataset_version_id"],
        split=row["split"],
        evaluated_at=_dt(row["evaluated_at"]),
        status=L8EvaluationStatus(row["status"]),
        sample_count=int(row["sample_count"]),
        eligible_samples=int(row["eligible_samples"]),
        excluded_samples=int(row["excluded_samples"]),
        metrics=json.loads(row["metrics_json"]),
        confusion_matrix=tuple(tuple(int(x) for x in item) for item in json.loads(row["confusion_matrix_json"])),
        class_distribution={str(k): int(v) for k, v in json.loads(row["class_distribution_json"]).items()},
        feature_schema_version=row["feature_schema_version"],
        label_version=row["label_version"],
        evaluation_config=json.loads(row["evaluation_config_json"]),
        evaluation_fingerprint=row["evaluation_fingerprint"],
        metadata=json.loads(row["metadata_json"]),
        windows=tuple(json.loads(row["windows_json"])),
        stability_summary=json.loads(row["stability_summary_json"]),
        started_at=None if row["started_at"] is None else _dt(row["started_at"]),
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


__all__ = ["L8EvaluationPersistenceError", "L8EvaluationRepository"]

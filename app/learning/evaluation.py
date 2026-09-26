"""L7 validation evaluation for registered model artifacts only."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app.learning.dataset import DatasetRow, DatasetSplit
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureStore
from app.learning.registry import EvaluationStatus, ModelEvaluation, ModelRegistryError, ModelVersion, ModelVersionStatus
from app.learning.training import LearningTrainingPipeline


class ModelEvaluationEngine:
    """Evaluate registered artifacts on VALIDATION only; OOS belongs to L8."""

    VERSION = "l7-evaluation-v1"

    def __init__(self, registry_repository: Any, dataset_repository: DatasetVersionRepository, feature_store: FeatureStore, training_pipeline: LearningTrainingPipeline):
        self.registry_repository = registry_repository
        self.dataset_repository = dataset_repository
        self.feature_store = feature_store
        self.training_pipeline = training_pipeline

    def evaluate(self, model_version_id: str, *, dataset_version_id: str | None = None, split: str = DatasetSplit.VALIDATION.value, evaluation_config: Mapping[str, Any] | None = None) -> ModelEvaluation:
        model = self.registry_repository.get(model_version_id)
        if model is None:
            raise ModelRegistryError(f"ModelVersion not found: {model_version_id}")
        if model.status == ModelVersionStatus.REVOKED:
            raise ModelRegistryError("revoked ModelVersion cannot be evaluated")
        if split != DatasetSplit.VALIDATION.value:
            raise ModelRegistryError("L7 evaluation is restricted to VALIDATION; OOS evaluation belongs to L8")
        dataset_id = dataset_version_id or model.dataset_version_id
        dataset = self.dataset_repository.get(dataset_id)
        if dataset is None:
            raise ModelRegistryError(f"DatasetVersion not found: {dataset_id}")
        if dataset.feature_schema_version != model.feature_schema_version:
            raise ModelRegistryError("DatasetVersion feature schema does not match ModelVersion")
        if dataset.label_version != model.label_version:
            raise ModelRegistryError("DatasetVersion label version does not match ModelVersion")

        rows = tuple(row for row in self.dataset_repository.list_rows(dataset_id, feature_store=self.feature_store) if row.split == DatasetSplit.VALIDATION)
        self._validate_rows(model, dataset, rows)
        fingerprint = _evaluation_fingerprint(model, dataset, rows, evaluation_config or {})
        evaluation_id = f"eval-{fingerprint[:32]}"
        existing = self.registry_repository.get_evaluation(evaluation_id)
        if existing is not None:
            return existing

        started = datetime.now(timezone.utc)
        running = ModelEvaluation(
            evaluation_id=evaluation_id,
            model_version_id=model.model_version_id,
            dataset_version_id=dataset.dataset_version_id,
            split=DatasetSplit.VALIDATION.value,
            evaluated_at=started,
            sample_count=len(rows),
            status=EvaluationStatus.RUNNING,
            metrics={},
            confusion_matrix=(),
            class_distribution={},
            feature_schema_version=model.feature_schema_version,
            label_version=model.label_version,
            evaluation_config=_canonical(evaluation_config or {}),
            evaluation_fingerprint=fingerprint,
            started_at=started,
        )
        self.registry_repository.create_evaluation(running)
        try:
            if not rows:
                raise ModelRegistryError("VALIDATION split is empty; evaluation cannot produce validation metrics")
            artifact_path = model.artifact_path
            artifact = self.training_pipeline.load_artifact(artifact_path, expected_sha256=model.artifact_sha256)
            self._validate_artifact(model, dataset, artifact)
            labels = tuple(row.outcome for row in rows)
            label_encoder = artifact["label_encoder"]
            classes = tuple(str(item) for item in label_encoder.classes_)
            unknown = sorted(set(labels) - set(classes))
            if unknown:
                raise ModelRegistryError(f"VALIDATION contains labels unseen by the registered artifact: {unknown}")
            predictions = self.training_pipeline.predict(artifact, rows)
            metrics, matrix, distribution = _classification_metrics(labels, predictions, classes)
            completed = datetime.now(timezone.utc)
            result = ModelEvaluation(
                evaluation_id=evaluation_id,
                model_version_id=model.model_version_id,
                dataset_version_id=dataset.dataset_version_id,
                split=DatasetSplit.VALIDATION.value,
                evaluated_at=completed,
                sample_count=len(rows),
                status=EvaluationStatus.COMPLETED,
                metrics=metrics,
                confusion_matrix=matrix,
                class_distribution=distribution,
                feature_schema_version=model.feature_schema_version,
                label_version=model.label_version,
                evaluation_config=_canonical(evaluation_config or {}),
                evaluation_fingerprint=fingerprint,
                started_at=started,
                completed_at=completed,
            )
            stored = self.registry_repository.mark_evaluation_completed(result)
            self.registry_repository.mark_evaluated(model.model_version_id)
            return stored
        except Exception as exc:
            failed = ModelEvaluation(
                evaluation_id=evaluation_id,
                model_version_id=model.model_version_id,
                dataset_version_id=dataset.dataset_version_id,
                split=DatasetSplit.VALIDATION.value,
                evaluated_at=datetime.now(timezone.utc),
                sample_count=len(rows),
                status=EvaluationStatus.FAILED,
                metrics={},
                confusion_matrix=(),
                class_distribution={},
                feature_schema_version=model.feature_schema_version,
                label_version=model.label_version,
                evaluation_config=_canonical(evaluation_config or {}),
                evaluation_fingerprint=fingerprint,
                started_at=started,
                completed_at=datetime.now(timezone.utc),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            self.registry_repository.mark_evaluation_failed(failed)
            raise

    @staticmethod
    def _validate_rows(model: ModelVersion, dataset: Any, rows: Sequence[DatasetRow]) -> None:
        for row in rows:
            if row.split != DatasetSplit.VALIDATION:
                raise ModelRegistryError("L7 evaluator received a non-VALIDATION row")
            if row.feature_schema_version != model.feature_schema_version:
                raise ModelRegistryError("validation feature schema mismatch")
            if row.label_version != model.label_version:
                raise ModelRegistryError("validation label version mismatch")
            if row.strategy_name != model.strategy_name or row.strategy_variant != model.strategy_variant or row.strategy_version != model.strategy_version:
                raise ModelRegistryError("validation strategy lineage mismatch")
            forbidden = {"outcome", "exit_price", "exit_timestamp", "exit_reason", "duration", "duration_seconds", "r_multiple", "pnl", "future_high", "future_low", "future_volatility", "future_regime"}
            for key in row.features.keys():
                lowered = str(key).lower()
                if lowered in forbidden or any(token in lowered for token in ("api_key", "password", "secret", "session", "token")):
                    raise ModelRegistryError(f"outcome/future/sensitive field present in evaluation feature set: {key}")

    @staticmethod
    def _validate_artifact(model: ModelVersion, dataset: Any, artifact: Mapping[str, Any]) -> None:
        expected = {
            "dataset_version_id": model.dataset_version_id if dataset.dataset_version_id == model.dataset_version_id else artifact.get("dataset_version_id"),
            "strategy_name": model.strategy_name,
            "strategy_variant": model.strategy_variant,
            "strategy_version": model.strategy_version,
            "model_type": model.model_type,
            "feature_schema_version": model.feature_schema_version,
            "label_version": model.label_version,
        }
        for key, value in expected.items():
            if artifact.get(key) != value:
                raise ModelRegistryError(f"registered artifact mismatch: {key}")
        if tuple(str(item) for item in artifact.get("feature_names", ())) != tuple(str(item) for item in artifact.get("feature_names", ())):
            raise ModelRegistryError("invalid feature ordering metadata")


def _classification_metrics(labels: Sequence[str], predictions: Sequence[str], classes: Sequence[str]) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...], dict[str, int]]:
    labels_list = list(classes)
    matrix = confusion_matrix(labels, predictions, labels=labels_list).tolist()
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
    }
    distribution = {str(label): int(sum(value == label for value in labels)) for label in labels_list}
    metrics["class_distribution"] = distribution
    metrics["sample_count"] = len(labels)
    return metrics, tuple(tuple(int(cell) for cell in row) for row in matrix), distribution


def _evaluation_fingerprint(model: ModelVersion, dataset: Any, rows: Sequence[DatasetRow], config: Mapping[str, Any]) -> str:
    payload = {
        "evaluation_version": ModelEvaluationEngine.VERSION,
        "model_version_id": model.model_version_id,
        "dataset_version_id": dataset.dataset_version_id,
        "split": DatasetSplit.VALIDATION.value,
        "feature_schema_version": model.feature_schema_version,
        "label_version": model.label_version,
        "evaluation_config": _canonical(config),
        "samples": [
            {"record_id": row.record_id, "row_fingerprint": row.row_fingerprint, "decision_timestamp": row.decision_timestamp.astimezone(timezone.utc).isoformat()}
            for row in sorted(rows, key=lambda item: (item.decision_timestamp, item.record_id))
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ModelRegistryError("evaluation configuration contains non-finite float")
    return value


__all__ = ["ModelEvaluationEngine"]

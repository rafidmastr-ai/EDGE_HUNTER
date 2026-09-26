"""L7 model registry contracts and artifact registration rules.

L7 deliberately separates TrainingRun, Model Artifact, and registered ModelVersion.
No production, champion/challenger, promotion, live inference, or model-selection
semantics belong here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from app.learning.dataset import DatasetVersion
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureStore
from app.learning.training import LearningTrainingPipeline, TrainingPipelineError, TrainingRun, TrainingStatus
from app.learning.training_repository import TrainingRunRepository


class ModelRegistryError(ValueError):
    """Base error for safe registry operations."""


class ModelVersionStatus(str, Enum):
    REGISTERED = "REGISTERED"
    EVALUATED = "EVALUATED"
    REVOKED = "REVOKED"


class EvaluationStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ModelVersion:
    model_version_id: str
    model_version_label: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    model_type: str
    training_run_id: str
    dataset_version_id: str
    feature_schema_version: str
    label_version: str
    artifact_path: str
    artifact_sha256: str
    artifact_format_version: str
    created_at: datetime
    registered_at: datetime
    status: ModelVersionStatus = ModelVersionStatus.REGISTERED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "model_version_id", "model_version_label", "strategy_name", "strategy_variant",
            "strategy_version", "model_type", "training_run_id", "dataset_version_id",
            "feature_schema_version", "label_version", "artifact_path", "artifact_sha256",
            "artifact_format_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ModelRegistryError(f"{name} must not be empty")
        for name in ("created_at", "registered_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ModelRegistryError(f"{name} must be timezone-aware")
        if self.registered_at < self.created_at:
            raise ModelRegistryError("registered_at must not precede created_at")
        _validate_json(self.metadata, "model_metadata")

    def lineage(self) -> dict[str, str]:
        return {
            "training_run_id": self.training_run_id,
            "dataset_version_id": self.dataset_version_id,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "model_type": self.model_type,
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version_id": self.model_version_id,
            "model_version_label": self.model_version_label,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "model_type": self.model_type,
            "training_run_id": self.training_run_id,
            "dataset_version_id": self.dataset_version_id,
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_format_version": self.artifact_format_version,
            "created_at": _iso(self.created_at),
            "registered_at": _iso(self.registered_at),
            "status": self.status.value,
            "metadata": _canonical(self.metadata),
        }


@dataclass(frozen=True)
class ModelEvaluation:
    evaluation_id: str
    model_version_id: str
    dataset_version_id: str
    split: str
    evaluated_at: datetime
    sample_count: int
    status: EvaluationStatus
    metrics: Mapping[str, Any]
    confusion_matrix: tuple[tuple[int, ...], ...]
    class_distribution: Mapping[str, int]
    feature_schema_version: str
    label_version: str
    evaluation_config: Mapping[str, Any]
    evaluation_fingerprint: str
    started_at: datetime
    completed_at: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id", "model_version_id", "dataset_version_id", "split",
            "feature_schema_version", "label_version", "evaluation_fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise ModelRegistryError(f"{name} must not be empty")
        for name in ("evaluated_at", "started_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ModelRegistryError(f"{name} must be timezone-aware")
        if self.completed_at is not None and (self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None):
            raise ModelRegistryError("completed_at must be timezone-aware")
        if self.sample_count < 0:
            raise ModelRegistryError("sample_count must be >= 0")
        _validate_json(self.metrics, "evaluation.metrics")
        _validate_json(self.class_distribution, "evaluation.class_distribution")
        _validate_json(self.evaluation_config, "evaluation.evaluation_config")
        if self.status == EvaluationStatus.COMPLETED and self.error_message is not None:
            raise ModelRegistryError("completed evaluation cannot contain error_message")
        if self.status == EvaluationStatus.FAILED and not self.error_message:
            raise ModelRegistryError("failed evaluation requires error_message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "model_version_id": self.model_version_id,
            "dataset_version_id": self.dataset_version_id,
            "split": self.split,
            "evaluated_at": _iso(self.evaluated_at),
            "sample_count": self.sample_count,
            "status": self.status.value,
            "metrics": _canonical(self.metrics),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "class_distribution": _canonical(self.class_distribution),
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "evaluation_config": _canonical(self.evaluation_config),
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class ModelRegistry:
    """Register completed L6 artifacts and validate lineage without promotion."""

    def __init__(
        self,
        repository: Any,
        training_repository: TrainingRunRepository,
        dataset_repository: DatasetVersionRepository,
        feature_store: FeatureStore,
        training_pipeline: LearningTrainingPipeline,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.training_repository = training_repository
        self.dataset_repository = dataset_repository
        self.feature_store = feature_store
        self.training_pipeline = training_pipeline
        self.artifact_root = Path(artifact_root) if artifact_root is not None else Path(training_pipeline.artifact_root)

    def register(self, training_run_id: str) -> ModelVersion:
        run = self.training_repository.get(training_run_id)
        if run is None:
            raise ModelRegistryError(f"TrainingRun not found: {training_run_id}")
        if run.status != TrainingStatus.COMPLETED:
            raise ModelRegistryError(f"cannot register TrainingRun with status {run.status.value}")
        dataset = self.dataset_repository.get(run.dataset_version_id)
        if dataset is None:
            raise ModelRegistryError("TrainingRun dataset lineage is missing")
        if not run.artifact_path or not run.artifact_sha256:
            raise ModelRegistryError("completed TrainingRun has no artifact metadata")

        artifact_path = self._safe_artifact_path(run.artifact_path)
        if not artifact_path.is_file():
            raise ModelRegistryError(f"model artifact not found: {artifact_path}")
        actual_sha = _file_sha256(artifact_path)
        if actual_sha != run.artifact_sha256:
            raise ModelRegistryError("model artifact checksum mismatch with TrainingRun")
        recorded_sha = run.artifact_metadata.get("sha256") if isinstance(run.artifact_metadata, Mapping) else None
        if recorded_sha is not None and recorded_sha != run.artifact_sha256:
            raise ModelRegistryError("TrainingRun artifact metadata checksum is inconsistent")

        try:
            artifact = self.training_pipeline.load_artifact(artifact_path, expected_sha256=run.artifact_sha256)
        except TrainingPipelineError as exc:
            raise ModelRegistryError(f"model artifact reload failed: {exc}") from exc
        self._validate_artifact_lineage(artifact, run, dataset)
        metadata = {
            "training_identity_hash": run.training_identity_hash,
            "training_parameters": _safe_mapping(run.training_config),
            "artifact_relative_path": _relative_path(artifact_path, self.artifact_root),
            "train_count": run.train_count,
            "validation_count": run.validation_count,
        }
        identity_payload = {
            "training_run_id": run.training_run_id,
            "dataset_version_id": run.dataset_version_id,
            "strategy_name": run.strategy_name,
            "strategy_variant": run.strategy_variant,
            "strategy_version": run.strategy_version,
            "model_type": run.model_type,
            "artifact_sha256": run.artifact_sha256,
            "artifact_format_version": artifact.get("artifact_format_version"),
            "feature_schema_version": run.feature_schema_version,
            "label_version": run.label_version,
        }
        identity_hash = _sha256(identity_payload)
        model_version_id = f"mdl-{identity_hash[:32]}"
        existing = self.repository.get(model_version_id)
        now = datetime.now(timezone.utc)
        candidate = ModelVersion(
            model_version_id=model_version_id,
            model_version_label=run.model_version_label,
            strategy_name=run.strategy_name,
            strategy_variant=run.strategy_variant,
            strategy_version=run.strategy_version,
            model_type=run.model_type,
            training_run_id=run.training_run_id,
            dataset_version_id=run.dataset_version_id,
            feature_schema_version=run.feature_schema_version,
            label_version=run.label_version,
            artifact_path=str(artifact_path.resolve()),
            artifact_sha256=run.artifact_sha256,
            artifact_format_version=str(artifact.get("artifact_format_version")),
            created_at=run.created_at,
            registered_at=now,
            status=ModelVersionStatus.REGISTERED if existing is None else existing.status,
            metadata=metadata,
        )
        if existing is not None:
            if _immutable_model_payload(existing) != _immutable_model_payload(candidate):
                raise ModelRegistryError("existing ModelVersion conflicts with requested registration")
            return existing
        return self.repository.create(candidate)

    def _safe_artifact_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        try:
            resolved = path.resolve(strict=False)
            root = self.artifact_root.expanduser().resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ModelRegistryError("artifact path is outside the configured training-artifact root") from exc
        return resolved

    @staticmethod
    def _validate_artifact_lineage(artifact: Mapping[str, Any], run: TrainingRun, dataset: DatasetVersion) -> None:
        pairs = {
            "training_identity_hash": run.training_identity_hash,
            "dataset_version_id": run.dataset_version_id,
            "strategy_name": run.strategy_name,
            "strategy_variant": run.strategy_variant,
            "strategy_version": run.strategy_version,
            "model_type": run.model_type,
            "model_version_label": run.model_version_label,
            "feature_schema_version": run.feature_schema_version,
            "label_version": run.label_version,
        }
        for key, expected in pairs.items():
            if artifact.get(key) != expected:
                raise ModelRegistryError(f"artifact metadata mismatch: {key}")
        if artifact.get("feature_schema_version") != dataset.feature_schema_version:
            raise ModelRegistryError("artifact feature schema does not match DatasetVersion")
        if artifact.get("label_version") != dataset.label_version:
            raise ModelRegistryError("artifact label version does not match DatasetVersion")
        required = {"model", "preprocessor", "label_encoder", "feature_names", "label_classes"}
        if not required.issubset(artifact):
            raise ModelRegistryError("artifact is missing required runtime components")


def _immutable_model_payload(model: ModelVersion) -> tuple[Any, ...]:
    return (
        model.model_version_label, model.strategy_name, model.strategy_variant, model.strategy_version,
        model.model_type, model.training_run_id, model.dataset_version_id, model.feature_schema_version,
        model.label_version, model.artifact_path, model.artifact_sha256, model.artifact_format_version,
        _canonical(model.metadata),
    )


def _safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    _validate_json(value, "metadata")
    return _canonical(value)


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelRegistryError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("password", "api_key", "secret", "session", "token", "subscription")):
                raise ModelRegistryError(f"{path} contains prohibited sensitive field: {key}")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise ModelRegistryError(f"{path} contains unsupported type: {type(value).__name__}")


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ModelRegistryError("non-finite value in canonical payload")
    return value


def _sha256(value: Any) -> str:
    payload = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


__all__ = [
    "EvaluationStatus",
    "ModelEvaluation",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelVersion",
    "ModelVersionStatus",
]

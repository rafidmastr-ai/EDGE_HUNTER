"""Offline, dataset-driven ML training pipeline for EDGE HUNTER L6.

L6 intentionally sits outside the live decision path.  It consumes an
immutable L5 DatasetVersion, trains only the reserved AI strategy scope, and
persists a TrainingRun plus a reloadable artifact.  No strategy, backtest,
feature-engine, outcome, API, or production integration is performed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.learning.dataset import DatasetRow, DatasetSplit, DatasetVersion
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureStore


DEFAULT_MODEL_TYPE = "LOGISTIC_REGRESSION"
DEFAULT_MODEL_VERSION_LABEL = "AI_V1"
DEFAULT_RANDOM_SEED = 42
TRAINING_ARTIFACT_FORMAT_VERSION = "l6-v1"
SUPPORTED_TRAINING_STRATEGIES = frozenset({"AI"})


class TrainingPipelineError(ValueError):
    """Base exception for safe, deterministic L6 failures."""


class TrainingStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingStage(str, Enum):
    VALIDATE = "VALIDATE"
    LOAD_DATASET = "LOAD_DATASET"
    PREPARE_FEATURES = "PREPARE_FEATURES"
    FIT = "FIT"
    VALIDATE_MODEL = "VALIDATE_MODEL"
    SAVE_ARTIFACT = "SAVE_ARTIFACT"
    PERSIST_RUN = "PERSIST_RUN"


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration that defines one reproducible training recipe."""

    dataset_version_id: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    model_type: str = DEFAULT_MODEL_TYPE
    model_version_label: str = DEFAULT_MODEL_VERSION_LABEL
    random_seed: int = DEFAULT_RANDOM_SEED
    training_parameters: Mapping[str, Any] = field(default_factory=dict)
    feature_schema_version: str = ""
    label_version: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_version_id",
            "strategy_name",
            "strategy_variant",
            "strategy_version",
            "model_type",
            "model_version_label",
            "feature_schema_version",
            "label_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise TrainingPipelineError(f"{field_name} must not be empty")
        if self.strategy_name.strip().upper() not in SUPPORTED_TRAINING_STRATEGIES:
            allowed = ", ".join(sorted(SUPPORTED_TRAINING_STRATEGIES))
            raise TrainingPipelineError(
                f"L6 training is restricted to reserved AI strategy scope; supported: {allowed}"
            )
        if self.model_type.strip().upper() != DEFAULT_MODEL_TYPE:
            raise TrainingPipelineError(f"unsupported model_type: {self.model_type}")
        if not isinstance(self.random_seed, int):
            raise TrainingPipelineError("random_seed must be an integer")
        _validate_json(self.training_parameters, "training_parameters", forbid_sensitive=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "model_type": self.model_type,
            "model_version_label": self.model_version_label,
            "random_seed": self.random_seed,
            "training_parameters": _canonical(self.training_parameters),
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
        }

    def identity_hash(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class TrainingRun:
    """Persisted execution record; unlike L5 datasets, runs may transition status."""

    training_run_id: str
    training_identity_hash: str
    dataset_version_id: str
    started_at: datetime
    completed_at: datetime | None
    status: TrainingStatus
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    model_type: str
    model_version_label: str
    feature_schema_version: str
    label_version: str
    train_count: int
    validation_count: int
    oos_count: int
    training_config: Mapping[str, Any]
    random_seed: int
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    artifact_metadata: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    error_stage: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _require_aware(self.started_at, "started_at")
        _require_aware(self.created_at, "created_at")
        if self.completed_at is not None:
            _require_aware(self.completed_at, "completed_at")
            if self.completed_at < self.started_at:
                raise TrainingPipelineError("completed_at must not precede started_at")
        for name in (
            "training_run_id",
            "training_identity_hash",
            "dataset_version_id",
            "strategy_name",
            "strategy_variant",
            "strategy_version",
            "model_type",
            "model_version_label",
            "feature_schema_version",
            "label_version",
        ):
            if not str(getattr(self, name)).strip():
                raise TrainingPipelineError(f"{name} must not be empty")
        if min(self.train_count, self.validation_count, self.oos_count) < 0:
            raise TrainingPipelineError("training split counts must be >= 0")
        _validate_json(self.training_config, "training_config", forbid_sensitive=True)
        _validate_json(self.metrics, "metrics", forbid_sensitive=True)
        _validate_json(self.artifact_metadata, "artifact_metadata", forbid_sensitive=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_run_id": self.training_run_id,
            "training_identity_hash": self.training_identity_hash,
            "dataset_version_id": self.dataset_version_id,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "status": self.status.value,
            "strategy_name": self.strategy_name,
            "strategy_variant": self.strategy_variant,
            "strategy_version": self.strategy_version,
            "model_type": self.model_type,
            "model_version_label": self.model_version_label,
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "oos_count": self.oos_count,
            "training_config": _canonical(self.training_config),
            "random_seed": self.random_seed,
            "metrics": _canonical(self.metrics),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_metadata": _canonical(self.artifact_metadata),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_stage": self.error_stage,
            "created_at": _iso(self.created_at),
        }


@dataclass(frozen=True)
class PreparedDataset:
    """Explicit separation of train/validation/OOS rows and matrices."""

    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    train_rows: tuple[DatasetRow, ...]
    validation_rows: tuple[DatasetRow, ...]
    oos_rows: tuple[DatasetRow, ...]
    x_train_raw: tuple[tuple[Any, ...], ...]
    y_train: tuple[str, ...]
    x_validation_raw: tuple[tuple[Any, ...], ...]
    y_validation: tuple[str, ...]
    provenance_distribution: Mapping[str, int]

    @property
    def train_rows_count(self) -> int:
        return len(self.x_train_raw)

    @property
    def validation_rows_count(self) -> int:
        return len(self.x_validation_raw)


@dataclass(frozen=True)
class TrainingResult:
    run: TrainingRun
    artifact_path: Path
    feature_names: tuple[str, ...]
    train_metrics: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    validation_unavailable: bool


class LearningTrainingPipeline:
    """Offline L6 trainer backed exclusively by an immutable DatasetVersion."""

    VERSION = TRAINING_ARTIFACT_FORMAT_VERSION

    def __init__(
        self,
        dataset_repository: DatasetVersionRepository,
        feature_store: FeatureStore,
        training_repository: Any,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.feature_store = feature_store
        self.training_repository = training_repository
        self.artifact_root = Path(artifact_root) if artifact_root is not None else (
            Path(__file__).resolve().parents[2] / "data" / "training_artifacts"
        )

    def train(self, config: TrainingConfig) -> TrainingResult:
        identity = config.identity_hash()
        run_id = f"trn-{uuid.uuid4().hex}"
        started = datetime.now(timezone.utc)

        # Validate the DatasetVersion before persisting a TrainingRun because
        # the database intentionally enforces a foreign key to immutable L5
        # datasets. A missing dataset is therefore a pre-run configuration
        # error, not a persisted training execution.
        stage = TrainingStage.LOAD_DATASET.value
        dataset = self._validate_dataset(config)

        run = TrainingRun(
            training_run_id=run_id,
            training_identity_hash=identity,
            dataset_version_id=config.dataset_version_id,
            started_at=started,
            completed_at=None,
            status=TrainingStatus.PENDING,
            strategy_name=config.strategy_name,
            strategy_variant=config.strategy_variant,
            strategy_version=config.strategy_version,
            model_type=config.model_type,
            model_version_label=config.model_version_label,
            feature_schema_version=config.feature_schema_version,
            label_version=config.label_version,
            train_count=0,
            validation_count=0,
            oos_count=0,
            training_config=config.to_dict(),
            random_seed=config.random_seed,
        )
        self.training_repository.create(run)
        try:
            self.training_repository.mark_running(run_id, started_at=started)

            stage = TrainingStage.PREPARE_FEATURES.value
            prepared = self._prepare_dataset(dataset, config)
            _assert_no_oos_in_training(prepared.train_rows, prepared.oos_rows)
            if prepared.train_rows_count == 0:
                raise TrainingPipelineError("TRAIN split is empty")

            stage = TrainingStage.FIT.value
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(list(prepared.y_train))
            if len(label_encoder.classes_) < 2:
                raise TrainingPipelineError("TRAIN split must contain at least two distinct labels")
            preprocessor = self._build_preprocessor(prepared)
            x_train = preprocessor.fit_transform(list(prepared.x_train_raw))
            model = self._fit_model(x_train, y_train_encoded, config)
            train_predictions = model.predict(x_train)
            train_metrics = _classification_metrics(
                y_train_encoded,
                train_predictions,
                label_encoder.classes_,
            )

            stage = TrainingStage.VALIDATE_MODEL.value
            validation_metrics: dict[str, Any] = {}
            validation_unavailable = prepared.validation_rows_count == 0
            if not validation_unavailable:
                unknown = sorted(set(prepared.y_validation) - set(label_encoder.classes_))
                if unknown:
                    raise TrainingPipelineError(
                        f"VALIDATION contains labels unseen in TRAIN: {unknown}"
                    )
                y_validation_encoded = label_encoder.transform(list(prepared.y_validation))
                x_validation = preprocessor.transform(list(prepared.x_validation_raw))
                validation_predictions = model.predict(x_validation)
                validation_metrics = _classification_metrics(
                    y_validation_encoded,
                    validation_predictions,
                    label_encoder.classes_,
                )
            metrics = {
                "train": train_metrics,
                "validation": validation_metrics,
                "validation_unavailable": validation_unavailable,
                "classes": [str(item) for item in label_encoder.classes_],
                "train_count": prepared.train_rows_count,
                "validation_count": prepared.validation_rows_count,
                "oos_count": len(prepared.oos_rows),
                "provenance_distribution": dict(prepared.provenance_distribution),
            }

            stage = TrainingStage.SAVE_ARTIFACT.value
            artifact_dir = self.artifact_root / run_id
            artifact_dir.mkdir(parents=True, exist_ok=False)
            artifact_path = artifact_dir / "training_artifact.joblib"
            artifact = {
                "artifact_format_version": self.VERSION,
                "model_type": config.model_type,
                "model_version_label": config.model_version_label,
                "training_identity_hash": identity,
                "dataset_version_id": config.dataset_version_id,
                "strategy_name": config.strategy_name,
                "strategy_variant": config.strategy_variant,
                "strategy_version": config.strategy_version,
                "feature_schema_version": config.feature_schema_version,
                "label_version": config.label_version,
                "random_seed": config.random_seed,
                "training_config": config.to_dict(),
                "feature_names": list(prepared.feature_names),
                "numeric_features": list(prepared.numeric_features),
                "categorical_features": list(prepared.categorical_features),
                "label_classes": [str(item) for item in label_encoder.classes_],
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
                "model": model,
            }
            dump(artifact, artifact_path, compress=3)
            artifact_sha256 = _file_sha256(artifact_path)
            artifact_metadata = {
                "format": self.VERSION,
                "relative_artifact": str(artifact_path.relative_to(self.artifact_root)),
                "sha256": artifact_sha256,
                "feature_count": len(prepared.feature_names),
                "classes": [str(item) for item in label_encoder.classes_],
            }

            stage = TrainingStage.PERSIST_RUN.value
            completed = datetime.now(timezone.utc)
            final_run = TrainingRun(
                training_run_id=run.training_run_id,
                training_identity_hash=run.training_identity_hash,
                dataset_version_id=run.dataset_version_id,
                started_at=run.started_at,
                completed_at=completed,
                status=TrainingStatus.COMPLETED,
                strategy_name=config.strategy_name,
                strategy_variant=config.strategy_variant,
                strategy_version=config.strategy_version,
                model_type=config.model_type,
                model_version_label=config.model_version_label,
                feature_schema_version=config.feature_schema_version,
                label_version=config.label_version,
                train_count=prepared.train_rows_count,
                validation_count=prepared.validation_rows_count,
                oos_count=len(prepared.oos_rows),
                training_config=config.to_dict(),
                random_seed=config.random_seed,
                metrics=metrics,
                artifact_path=str(artifact_path.resolve()),
                artifact_sha256=artifact_sha256,
                artifact_metadata=artifact_metadata,
                created_at=run.created_at,
            )
            self.training_repository.mark_completed(final_run)
            return TrainingResult(
                run=final_run,
                artifact_path=artifact_path.resolve(),
                feature_names=prepared.feature_names,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                validation_unavailable=validation_unavailable,
            )
        except Exception as exc:
            failed_at = datetime.now(timezone.utc)
            self.training_repository.mark_failed(
                run_id,
                completed_at=failed_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_stage=stage,
            )
            raise

    def load_artifact(self, path: str | Path, *, expected_sha256: str | None = None) -> Mapping[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise TrainingPipelineError(f"training artifact not found: {artifact_path}")
        if expected_sha256 is not None and _file_sha256(artifact_path) != expected_sha256:
            raise TrainingPipelineError("training artifact checksum mismatch")
        artifact = load(artifact_path)
        if not isinstance(artifact, Mapping):
            raise TrainingPipelineError("training artifact has invalid structure")
        if artifact.get("artifact_format_version") != self.VERSION:
            raise TrainingPipelineError("unsupported training artifact format version")
        required = {"model", "preprocessor", "label_encoder", "feature_names", "label_classes"}
        missing = sorted(required - set(artifact))
        if missing:
            raise TrainingPipelineError(f"training artifact missing fields: {missing}")
        return artifact

    def predict(self, artifact: Mapping[str, Any], rows: Sequence[DatasetRow]) -> tuple[str, ...]:
        self._validate_artifact_for_prediction(artifact)
        feature_names = tuple(str(item) for item in artifact["feature_names"])
        raw = _rows_to_matrix(rows, feature_names)
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]
        label_encoder = artifact["label_encoder"]
        transformed = preprocessor.transform(raw)
        encoded = model.predict(transformed)
        return tuple(str(item) for item in label_encoder.inverse_transform(encoded))

    def _validate_dataset(self, config: TrainingConfig) -> DatasetVersion:
        dataset = self.dataset_repository.get(config.dataset_version_id)
        if dataset is None:
            raise TrainingPipelineError(f"DatasetVersion not found: {config.dataset_version_id}")
        if dataset.feature_schema_version != config.feature_schema_version:
            raise TrainingPipelineError("feature_schema_version does not match DatasetVersion")
        if dataset.label_version != config.label_version:
            raise TrainingPipelineError("label_version does not match DatasetVersion")
        _validate_strategy_scope(dataset, config)
        return dataset

    def _prepare_dataset(self, dataset: DatasetVersion, config: TrainingConfig) -> PreparedDataset:
        rows = self.dataset_repository.list_rows(dataset.dataset_version_id, feature_store=self.feature_store)
        if len(rows) != dataset.row_count:
            raise TrainingPipelineError("DatasetVersion row count mismatch")
        ordered = tuple(sorted(rows, key=lambda item: (item.decision_timestamp, item.record_id)))
        for row in ordered:
            if row.feature_schema_version != dataset.feature_schema_version:
                raise TrainingPipelineError("dataset contains mixed feature schemas")
            if row.label_version != dataset.label_version:
                raise TrainingPipelineError("dataset contains mixed label versions")
            if row.strategy_name.upper() != config.strategy_name.upper():
                raise TrainingPipelineError("DatasetVersion contains a different strategy than the TrainingConfig")
            if row.strategy_variant != config.strategy_variant:
                raise TrainingPipelineError("DatasetVersion contains a different strategy variant than the TrainingConfig")
            if row.strategy_version != config.strategy_version:
                raise TrainingPipelineError("DatasetVersion contains a different strategy version than the TrainingConfig")
            _assert_features_safe(row)
        train = tuple(row for row in ordered if row.split == DatasetSplit.TRAIN)
        validation = tuple(row for row in ordered if row.split == DatasetSplit.VALIDATION)
        oos = tuple(row for row in ordered if row.split == DatasetSplit.OOS)
        _assert_no_oos_in_training(train, oos)
        if not train:
            feature_names = ()
            numeric = ()
            categorical = ()
        else:
            feature_names = _feature_names(train)
            if not feature_names:
                raise TrainingPipelineError("Dataset contains no usable feature columns in TRAIN")
            numeric, categorical = _feature_types(train, feature_names)
            expected_keys = set(feature_names)
            for split_name, split_rows in (("VALIDATION", validation), ("OOS", oos)):
                for row in split_rows:
                    keys = {str(key) for key in row.features.keys()}
                    if keys != expected_keys:
                        raise TrainingPipelineError(
                            f"{split_name} feature columns do not match TRAIN feature schema"
                        )
        provenance_counts: dict[str, int] = {}
        for row in ordered:
            key = row.source_type.value
            provenance_counts[key] = provenance_counts.get(key, 0) + 1
        return PreparedDataset(
            feature_names=feature_names,
            numeric_features=numeric,
            categorical_features=categorical,
            train_rows=train,
            validation_rows=validation,
            oos_rows=oos,
            x_train_raw=tuple(_rows_to_matrix(train, feature_names)),
            y_train=tuple(row.outcome for row in train),
            x_validation_raw=tuple(_rows_to_matrix(validation, feature_names)),
            y_validation=tuple(row.outcome for row in validation),
            provenance_distribution=dict(sorted(provenance_counts.items())),
        )

    def _build_preprocessor(self, prepared: PreparedDataset) -> ColumnTransformer:
        index = {name: idx for idx, name in enumerate(prepared.feature_names)}
        transformers: list[tuple[str, Pipeline, list[int]]] = []
        if prepared.numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    [index[name] for name in prepared.numeric_features],
                )
            )
        if prepared.categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    [index[name] for name in prepared.categorical_features],
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _fit_model(x_train: Any, y_train: Any, config: TrainingConfig) -> LogisticRegression:
        params = dict(config.training_parameters)
        allowed = {"C", "max_iter", "solver", "class_weight"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TrainingPipelineError(f"unsupported LogisticRegression parameters: {unknown}")
        params.setdefault("max_iter", 500)
        params.setdefault("solver", "lbfgs")
        params.setdefault("C", 1.0)
        params["random_state"] = config.random_seed
        model = LogisticRegression(**params)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def _validate_artifact_for_prediction(artifact: Mapping[str, Any]) -> None:
        if artifact.get("model_type") != DEFAULT_MODEL_TYPE:
            raise TrainingPipelineError("unsupported artifact model type")


def _validate_strategy_scope(dataset: DatasetVersion, config: TrainingConfig) -> None:
    rows_scope = dataset.filter_scope.get("strategies") if isinstance(dataset.filter_scope, Mapping) else None
    if rows_scope and config.strategy_name not in rows_scope:
        raise TrainingPipelineError("training strategy is not within DatasetVersion strategy scope")
    if not rows_scope:
        # Verify at least one persisted row carries the requested strategy later during matrix assembly.
        return


def _feature_names(rows: Sequence[DatasetRow]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(str(key) for key in row.features.keys())
    return tuple(sorted(names))


def _feature_types(rows: Sequence[DatasetRow], feature_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for name in feature_names:
        observed = [row.features.get(name) for row in rows if row.features.get(name) is not None]
        if not observed:
            raise TrainingPipelineError(f"feature has no observed values: {name}")
        kinds = {_value_kind(value) for value in observed}
        if kinds <= {"numeric"}:
            numeric.append(name)
        elif kinds <= {"categorical"}:
            categorical.append(name)
        else:
            raise TrainingPipelineError(f"feature has inconsistent value types: {name}")
    return tuple(numeric), tuple(categorical)


def _value_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "categorical"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return "numeric"
    if isinstance(value, str):
        return "categorical"
    raise TrainingPipelineError(f"unsupported feature value type: {type(value).__name__}")


def _rows_to_matrix(rows: Sequence[DatasetRow], feature_names: Sequence[str]) -> list[tuple[Any, ...]]:
    matrix: list[tuple[Any, ...]] = []
    for row in rows:
        _assert_features_safe(row)
        matrix.append(tuple(row.features.get(name) for name in feature_names))
    return matrix


def _assert_features_safe(row: DatasetRow) -> None:
    forbidden = {
        "outcome",
        "exit_price",
        "exit_timestamp",
        "exit_reason",
        "duration",
        "duration_seconds",
        "r_multiple",
        "pnl",
        "pnl_gross",
        "pnl_net",
        "future_high",
        "future_low",
        "future_volatility",
        "future_regime",
    }
    stack: list[tuple[str, Any]] = [(str(key), value) for key, value in row.features.items()]
    while stack:
        key, value = stack.pop()
        lowered = key.lower()
        if lowered in forbidden or any(token in lowered for token in ("api_key", "password", "secret", "token")):
            raise TrainingPipelineError(f"forbidden feature field: {key}")
        if isinstance(value, Mapping):
            stack.extend((f"{key}.{child}", child_value) for child, child_value in value.items())
        elif isinstance(value, (list, tuple)):
            raise TrainingPipelineError(f"non-scalar feature values are not supported: {key}")


def _assert_no_oos_in_training(train_rows: Sequence[Any], oos_rows: Sequence[Any]) -> None:
    contaminated = [getattr(row, "record_id", "<unknown>") for row in train_rows if getattr(row, "split", None) == DatasetSplit.OOS]
    if contaminated:
        raise TrainingPipelineError(f"OOS contamination detected in training matrix: {contaminated}")
    if not oos_rows:
        return
    train_ids = {getattr(row, "record_id", None) for row in train_rows}
    leaked_ids = sorted(train_ids.intersection(getattr(row, "record_id", None) for row in oos_rows))
    if leaked_ids:
        raise TrainingPipelineError(f"OOS record(s) present in training data: {leaked_ids}")


def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[Any]) -> dict[str, Any]:
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_distribution": {
            str(classes[index]): int(sum(item == index for item in y_true))
            for index in labels
        },
    }


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TrainingPipelineError(f"{name} must be timezone-aware")


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    _require_aware(parsed, "datetime")
    return parsed


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingPipelineError("non-finite training configuration value")
        return value
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


def _validate_json(value: Any, path: str, *, forbid_sensitive: bool) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TrainingPipelineError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrainingPipelineError(f"{path} contains a non-string key")
            lowered = key.lower()
            if forbid_sensitive and any(token in lowered for token in ("password", "api_key", "secret", "session", "token")):
                raise TrainingPipelineError(f"{path} contains prohibited sensitive field: {key}")
            _validate_json(item, f"{path}.{key}", forbid_sensitive=forbid_sensitive)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]", forbid_sensitive=forbid_sensitive)
        return
    raise TrainingPipelineError(f"{path} contains unsupported type: {type(value).__name__}")


__all__ = [
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_MODEL_VERSION_LABEL",
    "DEFAULT_RANDOM_SEED",
    "LearningTrainingPipeline",
    "PreparedDataset",
    "TrainingConfig",
    "TrainingPipelineError",
    "TrainingResult",
    "TrainingRun",
    "TrainingStage",
    "TrainingStatus",
]

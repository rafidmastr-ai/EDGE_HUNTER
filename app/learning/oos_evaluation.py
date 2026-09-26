"""L8 OOS, walk-forward, robustness, sensitivity, and stability evaluation.

L8 evaluates already-registered L6 model artifacts without fitting, retraining,
or mutating any training/dataset state.  OOS is transform-only and is never fed
back into training or model-selection logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app.learning.dataset import DatasetRow, DatasetSplit, DatasetVersion
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureStore
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.registry import ModelRegistryError, ModelVersion, ModelVersionStatus
from app.learning.training import LearningTrainingPipeline
from app.optimization.splits import make_walk_forward_folds


class L8EvaluationError(ValueError):
    """Raised when an L8 evaluation cannot be completed safely."""


class L8EvaluationType(str, Enum):
    OOS = "OOS"
    WALK_FORWARD = "WALK_FORWARD"
    ROBUSTNESS = "ROBUSTNESS"
    SENSITIVITY = "SENSITIVITY"


class L8EvaluationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class StabilitySummary:
    """Descriptive statistics only; no ranking or quality verdict is produced."""

    metric_summaries: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    window_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_summaries": {str(k): dict(v) for k, v in self.metric_summaries.items()},
            "window_count": self.window_count,
        }


@dataclass(frozen=True)
class L8EvaluationReport:
    """Persistable L8 evaluation report for one deterministic evaluation recipe."""

    evaluation_id: str
    evaluation_type: L8EvaluationType
    model_version_id: str
    dataset_version_id: str
    split: str
    evaluated_at: datetime
    status: L8EvaluationStatus
    sample_count: int
    eligible_samples: int
    excluded_samples: int
    metrics: Mapping[str, Any]
    confusion_matrix: tuple[tuple[int, ...], ...]
    class_distribution: Mapping[str, int]
    feature_schema_version: str
    label_version: str
    evaluation_config: Mapping[str, Any]
    evaluation_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    windows: tuple[Mapping[str, Any], ...] = ()
    stability_summary: Mapping[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id",
            "model_version_id",
            "dataset_version_id",
            "split",
            "feature_schema_version",
            "label_version",
            "evaluation_fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise L8EvaluationError(f"{name} must not be empty")
        for name in ("evaluated_at", "started_at", "completed_at"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise L8EvaluationError(f"{name} must be timezone-aware")
        if self.sample_count < 0 or self.eligible_samples < 0 or self.excluded_samples < 0:
            raise L8EvaluationError("sample counts must be >= 0")
        if self.eligible_samples > self.sample_count:
            raise L8EvaluationError("eligible_samples cannot exceed sample_count")
        _validate_json(self.metrics, "metrics")
        _validate_json(self.class_distribution, "class_distribution")
        _validate_json(self.evaluation_config, "evaluation_config")
        _validate_json(self.metadata, "metadata")
        _validate_json(self.stability_summary, "stability_summary")
        _validate_json(self.windows, "windows")
        if self.status == L8EvaluationStatus.FAILED and not self.error_message:
            raise L8EvaluationError("FAILED evaluation requires error_message")
        if self.status != L8EvaluationStatus.FAILED and self.error_message is not None:
            raise L8EvaluationError("only FAILED evaluations may carry error_message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_type": self.evaluation_type.value,
            "model_version_id": self.model_version_id,
            "dataset_version_id": self.dataset_version_id,
            "split": self.split,
            "evaluated_at": _iso(self.evaluated_at),
            "status": self.status.value,
            "sample_count": self.sample_count,
            "eligible_samples": self.eligible_samples,
            "excluded_samples": self.excluded_samples,
            "metrics": _canonical(self.metrics),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "class_distribution": _canonical(self.class_distribution),
            "feature_schema_version": self.feature_schema_version,
            "label_version": self.label_version,
            "evaluation_config": _canonical(self.evaluation_config),
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "metadata": _canonical(self.metadata),
            "windows": _canonical(self.windows),
            "stability_summary": _canonical(self.stability_summary),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class L8EvaluationEngine:
    """Evaluate a registered artifact on immutable L5 data without fitting."""

    VERSION = "l8-evaluation-v1"

    _FORBIDDEN_FEATURE_KEYS = {
        "outcome",
        "exit_price",
        "exit_timestamp",
        "exit_reason",
        "duration",
        "duration_seconds",
        "r_multiple",
        "pnl",
        "future_high",
        "future_low",
        "future_volatility",
        "future_regime",
    }
    _SENSITIVE_PARTS = ("password", "api_key", "secret", "session", "token", "subscription")

    def __init__(
        self,
        registry_repository: ModelRegistryRepository,
        dataset_repository: DatasetVersionRepository,
        feature_store: FeatureStore,
        training_pipeline: LearningTrainingPipeline,
        evaluation_repository: Any,
    ) -> None:
        self.registry_repository = registry_repository
        self.dataset_repository = dataset_repository
        self.feature_store = feature_store
        self.training_pipeline = training_pipeline
        self.evaluation_repository = evaluation_repository

    def evaluate_oos(
        self,
        model_version_id: str,
        *,
        dataset_version_id: str | None = None,
        evaluation_config: Mapping[str, Any] | None = None,
    ) -> L8EvaluationReport:
        config = _canonical(evaluation_config or {})
        model, dataset, rows = self._load_context(
            model_version_id,
            dataset_version_id=dataset_version_id,
            split=DatasetSplit.OOS,
            config=config,
        )
        fingerprint = self._fingerprint(
            L8EvaluationType.OOS,
            model,
            dataset,
            DatasetSplit.OOS.value,
            rows,
            config,
        )
        evaluation_id = f"l8-{L8EvaluationType.OOS.value.lower()}-{fingerprint[:32]}"
        existing = self.evaluation_repository.get(evaluation_id)
        if existing is not None:
            return existing

        started = datetime.now(timezone.utc)
        if not rows:
            return self._persist(
                self._report(
                    evaluation_id=evaluation_id,
                    evaluation_type=L8EvaluationType.OOS,
                    model=model,
                    dataset=dataset,
                    split=DatasetSplit.OOS.value,
                    status=L8EvaluationStatus.INSUFFICIENT_DATA,
                    sample_count=0,
                    eligible_samples=0,
                    excluded_samples=0,
                    metrics={},
                    matrix=(),
                    distribution={},
                    config=config,
                    fingerprint=fingerprint,
                    started=started,
                    completed=datetime.now(timezone.utc),
                    metadata={"reason": "NO_OOS_ROWS"},
                )
            )

        try:
            artifact = self._load_registered_artifact(model, dataset)
            self._validate_rows(model, dataset, rows)
            self._validate_artifact_feature_names(artifact, rows)
            labels = tuple(row.outcome for row in rows)
            self._validate_labels_against_artifact(artifact, labels)
            predictions = self._predict_transform_only(artifact, rows)
            metrics, matrix, distribution = _classification_metrics(
                labels,
                predictions,
                tuple(str(item) for item in artifact["label_encoder"].classes_),
            )
            metadata = {
                "sample_identity_hash": _sample_identity_hash(rows),
                "training_run_id": model.training_run_id,
                "source_distribution": _source_distribution(rows),
                "no_training_performed": True,
                "oos_feedback_loop": False,
            }
            report = self._report(
                evaluation_id=evaluation_id,
                evaluation_type=L8EvaluationType.OOS,
                model=model,
                dataset=dataset,
                split=DatasetSplit.OOS.value,
                status=L8EvaluationStatus.COMPLETED,
                sample_count=len(rows),
                eligible_samples=len(rows),
                excluded_samples=0,
                metrics=metrics,
                matrix=matrix,
                distribution=distribution,
                config=config,
                fingerprint=fingerprint,
                started=started,
                completed=datetime.now(timezone.utc),
                metadata=metadata,
            )
            return self._persist(report)
        except Exception as exc:
            report = self._failed_report(
                evaluation_id,
                L8EvaluationType.OOS,
                model,
                dataset,
                DatasetSplit.OOS.value,
                config,
                fingerprint,
                started,
                exc,
                sample_count=len(rows),
                metadata={"sample_identity_hash": _sample_identity_hash(rows)},
            )
            self._persist(report)
            raise

    def evaluate_walk_forward(
        self,
        model_version_id: str,
        *,
        dataset_version_id: str | None = None,
        folds: int = 3,
        min_train_rows: int | None = None,
        evaluation_config: Mapping[str, Any] | None = None,
    ) -> L8EvaluationReport:
        config = _canonical({
            **(evaluation_config or {}),
            "folds": folds,
            "min_train_rows": min_train_rows,
        })
        model, dataset, rows = self._load_context(
            model_version_id,
            dataset_version_id=dataset_version_id,
            split=DatasetSplit.OOS,
            config=config,
        )
        fingerprint = self._fingerprint(
            L8EvaluationType.WALK_FORWARD,
            model,
            dataset,
            DatasetSplit.OOS.value,
            rows,
            config,
        )
        evaluation_id = f"l8-{L8EvaluationType.WALK_FORWARD.value.lower()}-{fingerprint[:32]}"
        existing = self.evaluation_repository.get(evaluation_id)
        if existing is not None:
            return existing
        started = datetime.now(timezone.utc)
        if len(rows) < 3:
            return self._persist(self._report(
                evaluation_id=evaluation_id,
                evaluation_type=L8EvaluationType.WALK_FORWARD,
                model=model,
                dataset=dataset,
                split=DatasetSplit.OOS.value,
                status=L8EvaluationStatus.INSUFFICIENT_DATA,
                sample_count=len(rows),
                eligible_samples=0,
                excluded_samples=0,
                metrics={},
                matrix=(),
                distribution={},
                config=config,
                fingerprint=fingerprint,
                started=started,
                completed=datetime.now(timezone.utc),
                metadata={"reason": "INSUFFICIENT_OOS_ROWS_FOR_WALK_FORWARD"},
            ))
        try:
            artifact = self._load_registered_artifact(model, dataset)
            self._validate_rows(model, dataset, rows)
            self._validate_artifact_feature_names(artifact, rows)
            folds_data = make_walk_forward_folds(
                len(rows), folds=folds,
                min_train_bars=min_train_rows,
            )
            if not folds_data:
                return self._persist(self._report(
                    evaluation_id=evaluation_id,
                    evaluation_type=L8EvaluationType.WALK_FORWARD,
                    model=model,
                    dataset=dataset,
                    split=DatasetSplit.OOS.value,
                    status=L8EvaluationStatus.INSUFFICIENT_DATA,
                    sample_count=len(rows),
                    eligible_samples=0,
                    excluded_samples=0,
                    metrics={},
                    matrix=(),
                    distribution={},
                    config=config,
                    fingerprint=fingerprint,
                    started=started,
                    completed=datetime.now(timezone.utc),
                    metadata={"reason": "NO_WALK_FORWARD_FOLDS"},
                ))
            windows: list[dict[str, Any]] = []
            all_eval_rows: list[DatasetRow] = []
            all_predictions: list[str] = []
            all_labels: list[str] = []
            classes = tuple(str(item) for item in artifact["label_encoder"].classes_)
            for fold in folds_data:
                evaluation_rows = tuple(rows[fold.test_start:fold.test_end])
                if not evaluation_rows:
                    status = L8EvaluationStatus.INSUFFICIENT_DATA.value
                    windows.append({
                        "window_id": fold.fold_id,
                        "status": status,
                        "train_count": fold.train_end - fold.train_start,
                        "evaluation_count": 0,
                    })
                    continue
                predictions = self._predict_transform_only(artifact, evaluation_rows)
                labels = tuple(row.outcome for row in evaluation_rows)
                metrics, _, distribution = _classification_metrics(labels, predictions, classes)
                windows.append({
                    "window_id": fold.fold_id,
                    "status": L8EvaluationStatus.COMPLETED.value,
                    "train_start": _iso(rows[fold.train_start].decision_timestamp),
                    "train_end": _iso(rows[fold.train_end - 1].decision_timestamp),
                    "evaluation_start": _iso(evaluation_rows[0].decision_timestamp),
                    "evaluation_end": _iso(evaluation_rows[-1].decision_timestamp),
                    "train_count": fold.train_end - fold.train_start,
                    "evaluation_count": len(evaluation_rows),
                    "sample_identity_hash": _sample_identity_hash(evaluation_rows),
                    "metrics": metrics,
                    "class_distribution": distribution,
                })
                all_eval_rows.extend(evaluation_rows)
                all_predictions.extend(predictions)
                all_labels.extend(labels)
            if not all_eval_rows:
                status = L8EvaluationStatus.INSUFFICIENT_DATA
                metrics = {}
                matrix = ()
                distribution = {}
            else:
                status = L8EvaluationStatus.COMPLETED
                metrics, matrix, distribution = _classification_metrics(all_labels, all_predictions, classes)
            stability = _stability_summary(tuple(w["metrics"] for w in windows if "metrics" in w))
            metadata = {
                "window_count": len(windows),
                "completed_window_count": sum(w.get("status") == L8EvaluationStatus.COMPLETED.value for w in windows),
                "insufficient_window_count": sum(w.get("status") == L8EvaluationStatus.INSUFFICIENT_DATA.value for w in windows),
                "fixed_registered_artifact": True,
                "no_training_performed": True,
                "sample_identity_hash": _sample_identity_hash(tuple(all_eval_rows)),
            }
            return self._persist(self._report(
                evaluation_id=evaluation_id,
                evaluation_type=L8EvaluationType.WALK_FORWARD,
                model=model,
                dataset=dataset,
                split=DatasetSplit.OOS.value,
                status=status,
                sample_count=len(rows),
                eligible_samples=len(all_eval_rows),
                excluded_samples=len(rows) - len(all_eval_rows),
                metrics=metrics,
                matrix=matrix,
                distribution=distribution,
                config=config,
                fingerprint=fingerprint,
                started=started,
                completed=datetime.now(timezone.utc),
                metadata=metadata,
                windows=tuple(windows),
                stability_summary=stability.to_dict(),
            ))
        except Exception as exc:
            self._persist(self._failed_report(
                evaluation_id,
                L8EvaluationType.WALK_FORWARD,
                model,
                dataset,
                DatasetSplit.OOS.value,
                config,
                fingerprint,
                started,
                exc,
                sample_count=len(rows),
            ))
            raise

    def evaluate_robustness(
        self,
        model_version_id: str,
        *,
        dataset_version_id: str | None = None,
        minimum_sample: int = 1,
        include_symbol_cases: bool = True,
        include_timeframe_cases: bool = True,
        include_regime_cases: bool = True,
        include_temporal_windows: bool = True,
        folds: int = 3,
        evaluation_config: Mapping[str, Any] | None = None,
    ) -> L8EvaluationReport:
        config = _canonical({
            **(evaluation_config or {}),
            "minimum_sample": minimum_sample,
            "include_symbol_cases": include_symbol_cases,
            "include_timeframe_cases": include_timeframe_cases,
            "include_regime_cases": include_regime_cases,
            "include_temporal_windows": include_temporal_windows,
            "folds": folds,
        })
        if minimum_sample < 1:
            raise L8EvaluationError("minimum_sample must be >= 1")
        model, dataset, rows = self._load_context(
            model_version_id,
            dataset_version_id=dataset_version_id,
            split=DatasetSplit.OOS,
            config=config,
        )
        fingerprint = self._fingerprint(
            L8EvaluationType.ROBUSTNESS,
            model,
            dataset,
            DatasetSplit.OOS.value,
            rows,
            config,
        )
        evaluation_id = f"l8-{L8EvaluationType.ROBUSTNESS.value.lower()}-{fingerprint[:32]}"
        existing = self.evaluation_repository.get(evaluation_id)
        if existing is not None:
            return existing
        started = datetime.now(timezone.utc)
        if not rows:
            return self._persist(self._report(
                evaluation_id=evaluation_id,
                evaluation_type=L8EvaluationType.ROBUSTNESS,
                model=model,
                dataset=dataset,
                split=DatasetSplit.OOS.value,
                status=L8EvaluationStatus.INSUFFICIENT_DATA,
                sample_count=0,
                eligible_samples=0,
                excluded_samples=0,
                metrics={}, matrix=(), distribution={}, config=config,
                fingerprint=fingerprint, started=started,
                completed=datetime.now(timezone.utc),
                metadata={"reason": "NO_OOS_ROWS"},
            ))
        try:
            artifact = self._load_registered_artifact(model, dataset)
            self._validate_rows(model, dataset, rows)
            self._validate_artifact_feature_names(artifact, rows)
            classes = tuple(str(item) for item in artifact["label_encoder"].classes_)
            cases: list[dict[str, Any]] = []

            def add_case(case_id: str, case_type: str, case_rows: Sequence[DatasetRow]) -> None:
                ordered = tuple(sorted(case_rows, key=lambda r: (r.decision_timestamp, r.record_id)))
                if not ordered:
                    return
                if len(ordered) < minimum_sample:
                    cases.append({
                        "case_id": case_id,
                        "case_type": case_type,
                        "status": L8EvaluationStatus.INSUFFICIENT_DATA.value,
                        "sample_count": len(ordered),
                        "sample_identity_hash": _sample_identity_hash(ordered),
                    })
                    return
                predictions = self._predict_transform_only(artifact, ordered)
                labels = tuple(row.outcome for row in ordered)
                metrics, matrix, distribution = _classification_metrics(labels, predictions, classes)
                cases.append({
                    "case_id": case_id,
                    "case_type": case_type,
                    "status": L8EvaluationStatus.COMPLETED.value,
                    "sample_count": len(ordered),
                    "sample_identity_hash": _sample_identity_hash(ordered),
                    "metrics": metrics,
                    "confusion_matrix": [list(x) for x in matrix],
                    "class_distribution": distribution,
                })

            if include_temporal_windows:
                wf = make_walk_forward_folds(len(rows), folds=folds)
                for fold in wf:
                    add_case(
                        f"WINDOW:{fold.fold_id}",
                        "TEMPORAL_WINDOW",
                        rows[fold.test_start:fold.test_end],
                    )
            if include_symbol_cases:
                for symbol in sorted({row.symbol.upper() for row in rows}):
                    add_case(f"SYMBOL:{symbol}", "SYMBOL", tuple(r for r in rows if r.symbol.upper() == symbol))
            if include_timeframe_cases:
                for timeframe in sorted({row.timeframe for row in rows}):
                    add_case(f"TIMEFRAME:{timeframe}", "TIMEFRAME", tuple(r for r in rows if r.timeframe == timeframe))
            if include_regime_cases:
                regimes = sorted({str(row.market_regime) for row in rows if row.market_regime is not None})
                for regime in regimes:
                    add_case(f"REGIME:{regime}", "REGIME", tuple(r for r in rows if r.market_regime == regime))

            cases.sort(key=lambda item: item["case_id"])
            completed_cases = [case for case in cases if case["status"] == L8EvaluationStatus.COMPLETED.value]
            if not completed_cases:
                status = L8EvaluationStatus.INSUFFICIENT_DATA
                metrics = {}
                matrix = ()
                distribution = {}
            else:
                status = L8EvaluationStatus.COMPLETED
                # Robustness case metrics are descriptive; pooled metrics are on all
                # OOS rows and do not feed back into model training or selection.
                labels = tuple(row.outcome for row in rows)
                predictions = self._predict_transform_only(artifact, rows)
                metrics, matrix, distribution = _classification_metrics(labels, predictions, classes)
            stability = _stability_summary(tuple(case["metrics"] for case in completed_cases))
            metadata = {
                "case_count": len(cases),
                "completed_case_count": len(completed_cases),
                "insufficient_case_count": sum(case["status"] == L8EvaluationStatus.INSUFFICIENT_DATA.value for case in cases),
                "no_training_performed": True,
                "no_ranking": True,
                "no_model_selection": True,
            }
            return self._persist(self._report(
                evaluation_id=evaluation_id,
                evaluation_type=L8EvaluationType.ROBUSTNESS,
                model=model,
                dataset=dataset,
                split=DatasetSplit.OOS.value,
                status=status,
                sample_count=len(rows),
                eligible_samples=len(rows),
                excluded_samples=0,
                metrics=metrics,
                matrix=matrix,
                distribution=distribution,
                config=config,
                fingerprint=fingerprint,
                started=started,
                completed=datetime.now(timezone.utc),
                metadata=metadata,
                windows=tuple(cases),
                stability_summary=stability.to_dict(),
            ))
        except Exception as exc:
            self._persist(self._failed_report(
                evaluation_id,
                L8EvaluationType.ROBUSTNESS,
                model,
                dataset,
                DatasetSplit.OOS.value,
                config,
                fingerprint,
                started,
                exc,
                sample_count=len(rows),
            ))
            raise

    def evaluate_sensitivity(
        self,
        model_version_id: str,
        *,
        dataset_version_id: str | None = None,
        evaluation_config: Mapping[str, Any] | None = None,
    ) -> L8EvaluationReport:
        """Record applicability of model sensitivity without inventing retraining logic.

        L6's current LogisticRegression artifact has no runtime inference-parameter
        contract that can be varied without changing/retraining the artifact. L8
        therefore records NOT_APPLICABLE rather than fabricating parameter studies.
        """
        config = _canonical(evaluation_config or {})
        model, dataset, rows = self._load_context(
            model_version_id,
            dataset_version_id=dataset_version_id,
            split=DatasetSplit.OOS,
            config=config,
        )
        fingerprint = self._fingerprint(
            L8EvaluationType.SENSITIVITY,
            model,
            dataset,
            DatasetSplit.OOS.value,
            rows,
            config,
        )
        evaluation_id = f"l8-{L8EvaluationType.SENSITIVITY.value.lower()}-{fingerprint[:32]}"
        existing = self.evaluation_repository.get(evaluation_id)
        if existing is not None:
            return existing
        return self._persist(self._report(
            evaluation_id=evaluation_id,
            evaluation_type=L8EvaluationType.SENSITIVITY,
            model=model,
            dataset=dataset,
            split=DatasetSplit.OOS.value,
            status=L8EvaluationStatus.NOT_APPLICABLE,
            sample_count=len(rows),
            eligible_samples=0,
            excluded_samples=0,
            metrics={},
            matrix=(),
            distribution={},
            config=config,
            fingerprint=fingerprint,
            started=datetime.now(timezone.utc),
            completed=datetime.now(timezone.utc),
            metadata={
                "reason": "CURRENT_ARTIFACT_HAS_NO_RUNTIME_INFERENCE_PARAMETER_CONTRACT",
                "training_parameters_are_not_sensitivity_parameters": True,
                "no_retraining": True,
                "no_parameter_optimization": True,
            },
        ))

    def _load_context(
        self,
        model_version_id: str,
        *,
        dataset_version_id: str | None,
        split: DatasetSplit,
        config: Mapping[str, Any],
    ) -> tuple[ModelVersion, DatasetVersion, tuple[DatasetRow, ...]]:
        model = self.registry_repository.get(model_version_id)
        if model is None:
            raise ModelRegistryError(f"ModelVersion not found: {model_version_id}")
        if model.status == ModelVersionStatus.REVOKED:
            raise ModelRegistryError("revoked ModelVersion cannot be evaluated")
        if split != DatasetSplit.OOS:
            raise L8EvaluationError("L8 is restricted to OOS evaluation; validation evaluation belongs to L7")
        dataset_id = dataset_version_id or model.dataset_version_id
        dataset = self.dataset_repository.get(dataset_id)
        if dataset is None:
            raise L8EvaluationError(f"DatasetVersion not found: {dataset_id}")
        if dataset.feature_schema_version != model.feature_schema_version:
            raise L8EvaluationError("DatasetVersion feature schema does not match ModelVersion")
        if dataset.label_version != model.label_version:
            raise L8EvaluationError("DatasetVersion label version does not match ModelVersion")
        rows = tuple(
            row
            for row in self.dataset_repository.list_rows(dataset.dataset_version_id, feature_store=self.feature_store)
            if row.split == DatasetSplit.OOS
        )
        self._validate_rows(model, dataset, rows)
        return model, dataset, rows

    def _load_registered_artifact(self, model: ModelVersion, dataset: DatasetVersion) -> Mapping[str, Any]:
        try:
            artifact = self.training_pipeline.load_artifact(model.artifact_path, expected_sha256=model.artifact_sha256)
        except Exception as exc:
            raise L8EvaluationError(f"registered artifact reload failed: {exc}") from exc
        if artifact.get("dataset_version_id") != model.dataset_version_id:
            raise L8EvaluationError("registered artifact dataset lineage mismatch")
        for key in (
            "strategy_name", "strategy_variant", "strategy_version", "model_type",
            "feature_schema_version", "label_version",
        ):
            if artifact.get(key) != getattr(model, key):
                raise L8EvaluationError(f"registered artifact metadata mismatch: {key}")
        feature_names = tuple(str(item) for item in artifact.get("feature_names", ()))
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise L8EvaluationError("registered artifact has invalid feature schema")
        if not {"model", "preprocessor", "label_encoder"}.issubset(artifact):
            raise L8EvaluationError("registered artifact is missing runtime components")
        return artifact

    def _validate_rows(self, model: ModelVersion, dataset: DatasetVersion, rows: Sequence[DatasetRow]) -> None:
        seen: set[str] = set()
        expected_features: tuple[str, ...] | None = None
        for row in rows:
            if row.record_id in seen:
                raise L8EvaluationError(f"duplicate dataset row identity: {row.record_id}")
            seen.add(row.record_id)
            if row.split != DatasetSplit.OOS:
                raise L8EvaluationError("L8 received a non-OOS row")
            if row.feature_schema_version != model.feature_schema_version or row.feature_schema_version != dataset.feature_schema_version:
                raise L8EvaluationError("OOS feature schema mismatch")
            if row.label_version != model.label_version or row.label_version != dataset.label_version:
                raise L8EvaluationError("OOS label version mismatch")
            if row.strategy_name != model.strategy_name or row.strategy_variant != model.strategy_variant or row.strategy_version != model.strategy_version:
                raise L8EvaluationError("OOS strategy lineage mismatch")
            keys = tuple(sorted(str(key) for key in row.features.keys()))
            if expected_features is None:
                expected_features = keys
            elif keys != expected_features:
                raise L8EvaluationError("OOS feature names are not consistent across rows")
            for key in row.features:
                lowered = str(key).lower()
                if lowered in self._FORBIDDEN_FEATURE_KEYS or any(token in lowered for token in self._SENSITIVE_PARTS):
                    raise L8EvaluationError(f"outcome/future/sensitive field present in OOS features: {key}")

    def _validate_artifact_feature_names(self, artifact: Mapping[str, Any], rows: Sequence[DatasetRow]) -> None:
        expected = tuple(str(item) for item in artifact.get("feature_names", ()))
        actual = tuple(sorted(str(key) for key in rows[0].features.keys())) if rows else expected
        if set(expected) != set(actual):
            raise L8EvaluationError("evaluation feature columns do not match registered artifact feature schema")

    @staticmethod
    def _validate_labels_against_artifact(artifact: Mapping[str, Any], labels: Sequence[str]) -> None:
        classes = {str(item) for item in artifact["label_encoder"].classes_}
        unknown = sorted(set(labels) - classes)
        if unknown:
            raise L8EvaluationError(f"evaluation contains labels unseen by registered artifact: {unknown}")

    def _predict_transform_only(self, artifact: Mapping[str, Any], rows: Sequence[DatasetRow]) -> tuple[str, ...]:
        # LearningTrainingPipeline.predict() only calls preprocessor.transform()
        # and model.predict(); it never fits any preprocessing on the evaluation rows.
        try:
            return self.training_pipeline.predict(artifact, rows)
        except Exception as exc:
            raise L8EvaluationError(f"OOS prediction failed: {exc}") from exc

    def _fingerprint(
        self,
        evaluation_type: L8EvaluationType,
        model: ModelVersion,
        dataset: DatasetVersion,
        split: str,
        rows: Sequence[DatasetRow],
        config: Mapping[str, Any],
    ) -> str:
        payload = {
            "engine_version": self.VERSION,
            "evaluation_type": evaluation_type.value,
            "model_version_id": model.model_version_id,
            "dataset_version_id": dataset.dataset_version_id,
            "split": split,
            "feature_schema_version": model.feature_schema_version,
            "label_version": model.label_version,
            "evaluation_config": _canonical(config),
            "sample_identities": [
                {
                    "record_id": row.record_id,
                    "row_fingerprint": row.row_fingerprint,
                    "decision_timestamp": _iso(row.decision_timestamp),
                }
                for row in sorted(rows, key=lambda item: (item.decision_timestamp, item.record_id))
            ],
        }
        return _sha256(payload)

    def _report(
        self,
        *,
        evaluation_id: str,
        evaluation_type: L8EvaluationType,
        model: ModelVersion,
        dataset: DatasetVersion,
        split: str,
        status: L8EvaluationStatus,
        sample_count: int,
        eligible_samples: int,
        excluded_samples: int,
        metrics: Mapping[str, Any],
        matrix: Sequence[Sequence[int]],
        distribution: Mapping[str, int],
        config: Mapping[str, Any],
        fingerprint: str,
        started: datetime,
        completed: datetime,
        metadata: Mapping[str, Any] | None = None,
        windows: Sequence[Mapping[str, Any]] = (),
        stability_summary: Mapping[str, Any] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> L8EvaluationReport:
        report_metadata = {
            "strategy_name": model.strategy_name,
            "strategy_variant": model.strategy_variant,
            "strategy_version": model.strategy_version,
            "training_run_id": model.training_run_id,
            "model_type": model.model_type,
            "dataset_version_id": dataset.dataset_version_id,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_excluded_records": dataset.excluded_records,
            "dataset_exclusion_counts": dict(dataset.exclusion_counts),
            **dict(metadata or {}),
        }
        return L8EvaluationReport(
            evaluation_id=evaluation_id,
            evaluation_type=evaluation_type,
            model_version_id=model.model_version_id,
            dataset_version_id=dataset.dataset_version_id,
            split=split,
            evaluated_at=completed,
            status=status,
            sample_count=sample_count,
            eligible_samples=eligible_samples,
            excluded_samples=excluded_samples,
            metrics=_canonical(metrics),
            confusion_matrix=tuple(tuple(int(x) for x in row) for row in matrix),
            class_distribution={str(k): int(v) for k, v in distribution.items()},
            feature_schema_version=model.feature_schema_version,
            label_version=model.label_version,
            evaluation_config=_canonical(config),
            evaluation_fingerprint=fingerprint,
            metadata=_canonical(report_metadata),
            windows=tuple(_canonical(x) for x in windows),
            stability_summary=_canonical(stability_summary or {}),
            started_at=started,
            completed_at=completed,
            error_type=error_type,
            error_message=error_message,
        )

    def _failed_report(
        self,
        evaluation_id: str,
        evaluation_type: L8EvaluationType,
        model: ModelVersion,
        dataset: DatasetVersion,
        split: str,
        config: Mapping[str, Any],
        fingerprint: str,
        started: datetime,
        exc: Exception,
        *,
        sample_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> L8EvaluationReport:
        completed = datetime.now(timezone.utc)
        return self._report(
            evaluation_id=evaluation_id,
            evaluation_type=evaluation_type,
            model=model,
            dataset=dataset,
            split=split,
            status=L8EvaluationStatus.FAILED,
            sample_count=sample_count,
            eligible_samples=0,
            excluded_samples=0,
            metrics={},
            matrix=(),
            distribution={},
            config=config,
            fingerprint=fingerprint,
            started=started,
            completed=completed,
            metadata=metadata or {},
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def _persist(self, report: L8EvaluationReport) -> L8EvaluationReport:
        return self.evaluation_repository.create(report)


def _classification_metrics(
    labels: Sequence[str],
    predictions: Sequence[str],
    classes: Sequence[str],
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...], dict[str, int]]:
    if len(labels) != len(predictions):
        raise L8EvaluationError("labels/predictions length mismatch")
    labels_list = list(classes)
    matrix = confusion_matrix(labels, predictions, labels=labels_list).tolist()
    precision_values, recall_values, f1_values, _ = _per_class(labels, predictions, labels_list)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, labels=labels_list, average="macro", zero_division=0)),
        "per_class_precision": {c: float(v) for c, v in zip(labels_list, precision_values)},
        "per_class_recall": {c: float(v) for c, v in zip(labels_list, recall_values)},
        "per_class_f1": {c: float(v) for c, v in zip(labels_list, f1_values)},
    }
    distribution = {str(label): int(sum(value == label for value in labels)) for label in labels_list}
    metrics["class_distribution"] = distribution
    metrics["sample_count"] = len(labels)
    return metrics, tuple(tuple(int(cell) for cell in row) for row in matrix), distribution


def _per_class(labels: Sequence[str], predictions: Sequence[str], classes: Sequence[str]):
    return precision_score(labels, predictions, labels=list(classes), average=None, zero_division=0), recall_score(
        labels, predictions, labels=list(classes), average=None, zero_division=0
    ), f1_score(labels, predictions, labels=list(classes), average=None, zero_division=0), None


def _stability_summary(metric_rows: Sequence[Mapping[str, Any]]) -> StabilitySummary:
    numeric_names = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    output: dict[str, dict[str, float | None]] = {}
    for name in numeric_names:
        values = [float(row[name]) for row in metric_rows if name in row and isinstance(row[name], (int, float))]
        if not values:
            output[name] = {"mean": None, "median": None, "stddev": None, "min": None, "max": None}
        else:
            output[name] = {
                "mean": float(mean(values)),
                "median": float(median(values)),
                "stddev": float(pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
            }
    return StabilitySummary(metric_summaries=output, window_count=len(metric_rows))


def _sample_identity_hash(rows: Sequence[DatasetRow]) -> str:
    payload = [
        {
            "record_id": row.record_id,
            "row_fingerprint": row.row_fingerprint,
            "decision_timestamp": _iso(row.decision_timestamp),
        }
        for row in sorted(rows, key=lambda item: (item.decision_timestamp, item.record_id))
    ]
    return _sha256(payload)


def _source_distribution(rows: Sequence[DatasetRow]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = row.source_type.value
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise L8EvaluationError("non-finite value in evaluation payload")
    return value


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise L8EvaluationError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise L8EvaluationError(f"{path} contains unsupported type: {type(value).__name__}")


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _sha256(value: Any) -> str:
    payload = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "L8EvaluationEngine",
    "L8EvaluationError",
    "L8EvaluationReport",
    "L8EvaluationStatus",
    "L8EvaluationType",
    "StabilitySummary",
]

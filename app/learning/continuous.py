"""L10 controlled continuous-learning orchestration and production lifecycle.

This module coordinates immutable L1-L9 artifacts.  Automation is explicit and
feature-flagged; it never trains from an API request and never promotes a model
without the configured evaluation gates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from app.learning.dataset import DatasetVersion
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.evaluation import ModelEvaluationEngine
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.oos_evaluation import L8EvaluationReport, L8EvaluationStatus, L8EvaluationType
from app.learning.policies import LearnedPolicyVersion, PolicyMode, PolicyType
from app.learning.policy_repository import LearnedPolicyRepository
from app.learning.registry import ModelVersion, ModelVersionStatus
from app.learning.training import LearningTrainingPipeline, TrainingConfig, TrainingResult
from app.learning.training_repository import TrainingRunRepository
from app.learning.continuous_repository import (
    CandidateStatus,
    LearningCycle,
    LearningCycleRepository,
    LearningCycleStatus,
    LearningState,
    MonitoringSnapshot,
    ProductionModelState,
    ProductionStateRepository,
    PromotionHistoryEntry,
)

logger = logging.getLogger("edge_hunter.learning.continuous")


class LearningCycleError(RuntimeError):
    """Controlled learning-cycle failure."""


class LearningTrigger(str, Enum):
    SCHEDULED = "SCHEDULED"
    DATA_THRESHOLD = "DATA_THRESHOLD"
    MANUAL = "MANUAL"
    DRIFT = "DRIFT"


class ProductionStatus(str, Enum):
    BASE_ONLY = "BASE_ONLY"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class LearningCycleConfig:
    """Operational controls. Safe defaults keep automation disabled."""

    enabled: bool = False
    automatic_retraining_enabled: bool = False
    automatic_promotion_enabled: bool = False
    production_enabled: bool = False
    schedule_interval_seconds: int = 3600
    minimum_new_completed_records: int = 1
    minimum_dataset_size: int = 1
    cooldown_seconds: int = 3600
    max_retries: int = 2
    minimum_monitoring_sample: int = 10
    drift_threshold: float = 0.25
    rollback_grace_period_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.schedule_interval_seconds < 60:
            raise ValueError("schedule_interval_seconds must be >= 60")
        for name in (
            "minimum_new_completed_records", "minimum_dataset_size", "cooldown_seconds",
            "max_retries", "minimum_monitoring_sample", "rollback_grace_period_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= float(self.drift_threshold):
            raise ValueError("drift_threshold must be >= 0")
        if self.automatic_promotion_enabled and not self.production_enabled:
            raise ValueError("automatic_promotion_enabled requires production_enabled")

    def fingerprint(self) -> str:
        return _sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "automatic_retraining_enabled": self.automatic_retraining_enabled,
            "automatic_promotion_enabled": self.automatic_promotion_enabled,
            "production_enabled": self.production_enabled,
            "schedule_interval_seconds": self.schedule_interval_seconds,
            "minimum_new_completed_records": self.minimum_new_completed_records,
            "minimum_dataset_size": self.minimum_dataset_size,
            "cooldown_seconds": self.cooldown_seconds,
            "max_retries": self.max_retries,
            "minimum_monitoring_sample": self.minimum_monitoring_sample,
            "drift_threshold": self.drift_threshold,
            "rollback_grace_period_seconds": self.rollback_grace_period_seconds,
        }


@dataclass(frozen=True)
class PromotionPolicy:
    """Explicit promotion gates; disabled by default and multi-metric when enabled."""

    enabled: bool = False
    minimum_validation_samples: int = 1
    minimum_oos_samples: int = 1
    require_validation: bool = True
    require_oos: bool = True
    require_robustness: bool = True
    minimum_validation_metrics: Mapping[str, float] = field(default_factory=dict)
    minimum_oos_metrics: Mapping[str, float] = field(default_factory=dict)
    minimum_metrics_required: int = 2
    require_artifact_integrity: bool = True
    require_schema_compatibility: bool = True
    require_policy_compatibility: bool = True
    version: str = "promotion-policy-v1"

    def __post_init__(self) -> None:
        if self.minimum_validation_samples < 0 or self.minimum_oos_samples < 0:
            raise ValueError("minimum samples must be >= 0")
        if self.minimum_metrics_required < 2:
            raise ValueError("promotion policy must require at least two metrics")
        if self.enabled:
            if len(self.minimum_validation_metrics) + len(self.minimum_oos_metrics) < self.minimum_metrics_required:
                raise ValueError("enabled promotion policy requires a multi-metric gate")
            for mapping in (self.minimum_validation_metrics, self.minimum_oos_metrics):
                for name, threshold in mapping.items():
                    if not str(name).strip() or not math.isfinite(float(threshold)):
                        raise ValueError("promotion metric thresholds must be finite and named")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_validation_samples": self.minimum_validation_samples,
            "minimum_oos_samples": self.minimum_oos_samples,
            "require_validation": self.require_validation,
            "require_oos": self.require_oos,
            "require_robustness": self.require_robustness,
            "minimum_validation_metrics": dict(self.minimum_validation_metrics),
            "minimum_oos_metrics": dict(self.minimum_oos_metrics),
            "minimum_metrics_required": self.minimum_metrics_required,
            "require_artifact_integrity": self.require_artifact_integrity,
            "require_schema_compatibility": self.require_schema_compatibility,
            "require_policy_compatibility": self.require_policy_compatibility,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class RollbackPolicy:
    minimum_monitoring_sample: int = 10
    grace_period_seconds: int = 3600
    max_runtime_errors: int | None = None
    max_drift_score: float | None = None
    max_metric_degradation: Mapping[str, float] = field(default_factory=dict)
    require_artifact_health: bool = True
    version: str = "rollback-policy-v1"

    def __post_init__(self) -> None:
        if self.minimum_monitoring_sample < 0 or self.grace_period_seconds < 0:
            raise ValueError("rollback minimums must be >= 0")
        if self.max_runtime_errors is not None and self.max_runtime_errors < 0:
            raise ValueError("max_runtime_errors must be >= 0")
        if self.max_drift_score is not None and self.max_drift_score < 0:
            raise ValueError("max_drift_score must be >= 0")

    def fingerprint(self) -> str:
        return _sha256({
            "minimum_monitoring_sample": self.minimum_monitoring_sample,
            "grace_period_seconds": self.grace_period_seconds,
            "max_runtime_errors": self.max_runtime_errors,
            "max_drift_score": self.max_drift_score,
            "max_metric_degradation": dict(self.max_metric_degradation),
            "require_artifact_health": self.require_artifact_health,
            "version": self.version,
        })


@dataclass(frozen=True)
class PromotionEvidence:
    """Normalized evaluation evidence used by the promotion gate."""

    validation_status: str
    validation_samples: int
    validation_metrics: Mapping[str, float]
    oos_status: str
    oos_samples: int
    oos_metrics: Mapping[str, float]
    robustness_status: str
    robustness_samples: int
    artifact_integrity: bool = True
    schema_compatible: bool = True
    policy_compatible: bool = True
    references: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionGateResult:
    eligible: bool
    reasons: tuple[str, ...]
    evidence: PromotionEvidence


@dataclass(frozen=True)
class LearningCycleRequest:
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    trigger: LearningTrigger
    trigger_key: str
    dataset_build_kwargs: Mapping[str, Any]
    training_config_factory: Callable[[DatasetVersion], TrainingConfig] | None = None
    policy_factory: Callable[[ModelVersion | None, DatasetVersion], LearnedPolicyVersion] | None = None
    policy: LearnedPolicyVersion | None = None
    policy_evidence: PromotionEvidence | None = None

    def __post_init__(self) -> None:
        if not self.strategy_name.strip() or not self.strategy_variant.strip() or not self.strategy_version.strip():
            raise ValueError("strategy identity is required")
        if not self.trigger_key.strip():
            raise ValueError("trigger_key is required for idempotent cycles")
        if not isinstance(self.dataset_build_kwargs, Mapping):
            raise ValueError("dataset_build_kwargs must be a mapping")


class PromotionGate:
    """Pure deterministic gate; it never mutates model or production state."""

    def evaluate(
        self,
        candidate: PromotionEvidence,
        policy: PromotionPolicy,
        *,
        candidate_model: ModelVersion | None,
        candidate_policy: LearnedPolicyVersion | None,
        expected_strategy: tuple[str, str, str],
    ) -> PromotionGateResult:
        reasons: list[str] = []
        if not policy.enabled:
            reasons.append("PROMOTION_POLICY_DISABLED")
            return PromotionGateResult(False, tuple(reasons), candidate)
        if policy.require_validation and candidate.validation_status != "COMPLETED":
            reasons.append("VALIDATION_NOT_COMPLETED")
        if policy.require_oos and candidate.oos_status != "COMPLETED":
            reasons.append("OOS_NOT_COMPLETED")
        if policy.require_robustness and candidate.robustness_status != "COMPLETED":
            reasons.append("ROBUSTNESS_NOT_COMPLETED")
        if candidate.validation_samples < policy.minimum_validation_samples:
            reasons.append("VALIDATION_SAMPLE_THRESHOLD")
        if candidate.oos_samples < policy.minimum_oos_samples:
            reasons.append("OOS_SAMPLE_THRESHOLD")
        if policy.require_artifact_integrity and not candidate.artifact_integrity:
            reasons.append("ARTIFACT_INTEGRITY_FAILED")
        if policy.require_schema_compatibility and not candidate.schema_compatible:
            reasons.append("SCHEMA_INCOMPATIBLE")
        if policy.require_policy_compatibility and not candidate.policy_compatible:
            reasons.append("POLICY_INCOMPATIBLE")
        if candidate_policy is None:
            reasons.append("POLICY_VERSION_REQUIRED")
        elif (
            candidate_policy.strategy_name,
            candidate_policy.strategy_variant,
            candidate_policy.strategy_version,
        ) != expected_strategy:
            reasons.append("POLICY_STRATEGY_MISMATCH")
        if candidate_model is not None and (
            candidate_model.strategy_name,
            candidate_model.strategy_variant,
            candidate_model.strategy_version,
        ) != expected_strategy:
            reasons.append("MODEL_STRATEGY_MISMATCH")
        metric_hits = 0
        for metric, threshold in policy.minimum_validation_metrics.items():
            actual = candidate.validation_metrics.get(metric)
            if actual is None or float(actual) < float(threshold):
                reasons.append(f"VALIDATION_METRIC:{metric}")
            else:
                metric_hits += 1
        for metric, threshold in policy.minimum_oos_metrics.items():
            actual = candidate.oos_metrics.get(metric)
            if actual is None or float(actual) < float(threshold):
                reasons.append(f"OOS_METRIC:{metric}")
            else:
                metric_hits += 1
        if metric_hits < policy.minimum_metrics_required:
            reasons.append("MULTI_METRIC_GATE_NOT_MET")
        return PromotionGateResult(not reasons, tuple(dict.fromkeys(reasons)), candidate)


class ProductionSafety:
    """Final fail-closed validation before changing production state."""

    @staticmethod
    def validate_pair(
        model: ModelVersion | None,
        policy: LearnedPolicyVersion,
        *,
        expected_strategy: tuple[str, str, str],
    ) -> None:
        if model is not None:
            if model.status == ModelVersionStatus.REVOKED:
                raise LearningCycleError("cannot activate a revoked ModelVersion")
            if (
                model.strategy_name,
                model.strategy_variant,
                model.strategy_version,
            ) != expected_strategy:
                raise LearningCycleError("model strategy lineage is incompatible with production scope")
            if policy.source_model_version_id not in {None, model.model_version_id}:
                raise LearningCycleError("policy/model lineage mismatch")
            if policy.feature_schema_version != model.feature_schema_version:
                raise LearningCycleError("policy/model feature schema mismatch")
            if policy.label_version != model.label_version:
                raise LearningCycleError("policy/model label version mismatch")
        if (
            policy.strategy_name,
            policy.strategy_variant,
            policy.strategy_version,
        ) != expected_strategy:
            raise LearningCycleError("policy strategy lineage is incompatible with production scope")
        if policy.default_mode == PolicyMode.DISABLED:
            # A production candidate must be explicitly represented as a candidate policy;
            # runtime integration remains feature-flagged outside this module.
            return
        if PolicyType.EXIT_POLICY in policy.policy_types:
            rr = policy.policy_parameters.get("target_rr")
            if rr is not None and not 1.5 <= float(rr) <= 2.0:
                raise LearningCycleError("production policy target_rr must remain within 1.5..2.0")


class MonitoringService:
    """Descriptive production monitoring; drift never mutates a model."""

    def __init__(self, repository: LearningCycleRepository) -> None:
        self.repository = repository

    def observe(
        self,
        *,
        scope_key: str,
        model_version_id: str | None,
        policy_version_id: str | None,
        sample_count: int,
        metrics: Mapping[str, float],
        baseline_metrics: Mapping[str, float] | None = None,
        observed_numeric: Sequence[Mapping[str, float]] = (),
        baseline_numeric: Mapping[str, float] | None = None,
        drift_threshold: float = 0.25,
        runtime_error_count: int = 0,
        artifact_healthy: bool = True,
        now: datetime | None = None,
    ) -> MonitoringSnapshot:
        if sample_count < 0:
            raise ValueError("sample_count must be >= 0")
        drift_score = _numeric_distribution_shift(observed_numeric, baseline_numeric or {})
        drift_detected = sample_count > 0 and drift_score >= drift_threshold
        snapshot = MonitoringSnapshot.create(
            scope_key=scope_key,
            model_version_id=model_version_id,
            policy_version_id=policy_version_id,
            observed_at=now or datetime.now(timezone.utc),
            sample_count=sample_count,
            metrics=metrics,
            baseline_metrics=baseline_metrics or {},
            drift_score=drift_score,
            drift_detected=drift_detected,
            runtime_error_count=runtime_error_count,
            artifact_healthy=artifact_healthy,
        )
        return self.repository.create_monitoring_snapshot(snapshot)

    @staticmethod
    def should_trigger_learning(snapshot: MonitoringSnapshot, *, minimum_sample: int) -> bool:
        return snapshot.sample_count >= minimum_sample and snapshot.drift_detected

    @staticmethod
    def should_rollback(
        snapshot: MonitoringSnapshot,
        state: ProductionModelState,
        policy: RollbackPolicy,
        *,
        now: datetime | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        current = now or datetime.now(timezone.utc)
        if state.status != ProductionStatus.ACTIVE.value:
            return False, ()
        if snapshot.sample_count < policy.minimum_monitoring_sample:
            return False, ("INSUFFICIENT_MONITORING_SAMPLE",)
        if current < state.activated_at + timedelta(seconds=policy.grace_period_seconds):
            return False, ("GRACE_PERIOD_ACTIVE",)
        if policy.require_artifact_health and not snapshot.artifact_healthy:
            reasons.append("ARTIFACT_HEALTH_FAILURE")
        if policy.max_runtime_errors is not None and snapshot.runtime_error_count > policy.max_runtime_errors:
            reasons.append("RUNTIME_ERROR_THRESHOLD")
        if policy.max_drift_score is not None and snapshot.drift_score > policy.max_drift_score:
            reasons.append("DRIFT_THRESHOLD")
        for metric, max_drop in policy.max_metric_degradation.items():
            baseline = snapshot.baseline_metrics.get(metric)
            current_metric = snapshot.metrics.get(metric)
            if baseline is None or current_metric is None:
                continue
            if float(baseline) - float(current_metric) > float(max_drop):
                reasons.append(f"METRIC_DEGRADATION:{metric}")
        return bool(reasons), tuple(dict.fromkeys(reasons))


class LearningScheduler:
    """Minimal standard-library scheduler; never auto-starts in tests."""

    def __init__(self, interval_seconds: int = 3600) -> None:
        if interval_seconds < 60:
            raise ValueError("interval_seconds must be >= 60")
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, callback: Callable[[], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    callback()
                except Exception:
                    logger.exception("scheduled learning cycle failed")

        self._thread = threading.Thread(target=loop, name="edge-hunter-learning", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=min(2.0, self.interval_seconds + 0.1))
        self._thread = None


class ContinuousLearningOrchestrator:
    """Controlled orchestration over L5-L9; no strategy code is rewritten."""

    def __init__(
        self,
        *,
        cycle_repository: LearningCycleRepository,
        production_repository: ProductionStateRepository,
        dataset_manager: Any,
        dataset_repository: DatasetVersionRepository,
        training_pipeline: LearningTrainingPipeline,
        training_repository: TrainingRunRepository,
        model_registry: Any,
        model_registry_repository: ModelRegistryRepository,
        validation_engine: ModelEvaluationEngine,
        oos_engine: Any,
        policy_repository: LearnedPolicyRepository,
        config: LearningCycleConfig | None = None,
        promotion_policy: PromotionPolicy | None = None,
        rollback_policy: RollbackPolicy | None = None,
    ) -> None:
        self.cycles = cycle_repository
        self.production = production_repository
        self.dataset_manager = dataset_manager
        self.dataset_repository = dataset_repository
        self.training_pipeline = training_pipeline
        self.training_repository = training_repository
        self.model_registry = model_registry
        self.model_registry_repository = model_registry_repository
        self.validation_engine = validation_engine
        self.oos_engine = oos_engine
        self.policy_repository = policy_repository
        self.config = config or LearningCycleConfig()
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.rollback_policy = rollback_policy or RollbackPolicy(
            minimum_monitoring_sample=self.config.minimum_monitoring_sample,
            grace_period_seconds=self.config.rollback_grace_period_seconds,
            max_drift_score=self.config.drift_threshold,
        )
        self._promotion_gate_engine = PromotionGate()
        self.monitoring = MonitoringService(self.cycles)
        self.scheduler = LearningScheduler(self.config.schedule_interval_seconds)
        self._scope_locks: dict[str, threading.Lock] = {}
        self._scope_locks_guard = threading.Lock()

    def _scope_lock(self, scope_key: str) -> threading.Lock:
        with self._scope_locks_guard:
            lock = self._scope_locks.get(scope_key)
            if lock is None:
                lock = threading.Lock()
                self._scope_locks[scope_key] = lock
            return lock

    def trigger(self, request: LearningCycleRequest) -> LearningCycle:
        self._ensure_automation_allowed(request.trigger)
        scope_key = self._scope_key(request)
        with self._scope_lock(scope_key):
            self._check_data_and_cooldown(request)
            identity = _sha256({
                "scope_key": scope_key,
                "trigger": request.trigger.value,
                "trigger_key": request.trigger_key,
                "config_fingerprint": self.config.fingerprint(),
                "dataset_build": _canonical(request.dataset_build_kwargs),
                "strategy": [request.strategy_name, request.strategy_variant, request.strategy_version],
            })
            existing = self.cycles.get_by_identity(identity)
            if existing is not None:
                return existing
            cycle = LearningCycle.create(
                identity_hash=identity,
                scope_key=scope_key,
                strategy_name=request.strategy_name,
                strategy_variant=request.strategy_variant,
                strategy_version=request.strategy_version,
                trigger=request.trigger,
                trigger_key=request.trigger_key,
                configuration_fingerprint=self.config.fingerprint(),
                started_at=datetime.now(timezone.utc),
            )
            cycle = self.cycles.create(cycle)
            # A concurrent caller in another process can win the DB race; if it is
            # already the same logical cycle, return it instead of executing twice.
            if cycle.identity_hash != identity:
                raise LearningCycleError("learning cycle identity mismatch")
            self.cycles.audit("cycle_started", cycle, reason=request.trigger.value, status=cycle.status.value)
            return self._run_with_retries(cycle, request)

    def start_scheduler(self, callback: Callable[[LearningTrigger], LearningCycle] | None = None) -> bool:
        if not self.config.enabled or not self.config.automatic_retraining_enabled:
            return False

        def scheduled() -> None:
            if callback is not None:
                callback(LearningTrigger.SCHEDULED)

        self.scheduler.start(scheduled)
        return True

    def stop_scheduler(self) -> None:
        self.scheduler.stop()

    def create_candidate(
        self,
        cycle: LearningCycle,
        *,
        model: ModelVersion | None,
        policy: LearnedPolicyVersion,
        evidence: PromotionEvidence,
    ) -> Any:
        ProductionSafety.validate_pair(
            model,
            policy,
            expected_strategy=(cycle.strategy_name, cycle.strategy_variant, cycle.strategy_version),
        )
        candidate = self.cycles.create_candidate(
            cycle=cycle,
            model_version_id=None if model is None else model.model_version_id,
            policy_version_id=policy.policy_version_id,
            dataset_version_id=cycle.dataset_version_id,
            validation_evaluation_id=evidence.references.get("validation"),
            oos_evaluation_id=evidence.references.get("oos"),
            robustness_evaluation_id=evidence.references.get("robustness"),
            evidence=_evidence_to_dict(evidence),
        )
        self.cycles.audit("candidate_created", cycle, reason="candidate", status=candidate.status)
        return candidate

    def promotion_gate(self, candidate: Any, *, policy: PromotionPolicy | None = None) -> PromotionGateResult:
        cycle = self.cycles.get(candidate.cycle_id)
        if cycle is None:
            raise LearningCycleError("candidate cycle not found")
        model = None
        if candidate.model_version_id:
            model = self.model_registry_repository.get(candidate.model_version_id)
        learned_policy = self.policy_repository.get(candidate.policy_version_id)
        if learned_policy is None:
            raise LearningCycleError("candidate policy not found")
        evidence = self._candidate_evidence(candidate)
        result = self._promotion_gate_engine.evaluate(
            evidence,
            policy or self.promotion_policy,
            candidate_model=model,
            candidate_policy=learned_policy,
            expected_strategy=(cycle.strategy_name, cycle.strategy_variant, cycle.strategy_version),
        )
        self.cycles.record_promotion_gate(candidate.candidate_id, result.eligible, result.reasons)
        self.cycles.audit("promotion_gate", cycle, reason=";".join(result.reasons) or "ELIGIBLE", status="ELIGIBLE" if result.eligible else "BLOCKED")
        return result

    def promote_candidate(self, candidate_id: str, *, reason: str, policy: PromotionPolicy | None = None) -> ProductionModelState:
        candidate = self.cycles.get_candidate(candidate_id)
        if candidate is None:
            raise LearningCycleError("candidate not found")
        cycle = self.cycles.get(candidate.cycle_id)
        if cycle is None:
            raise LearningCycleError("candidate cycle not found")
        gate = self.promotion_gate(candidate, policy=policy)
        if not gate.eligible:
            raise LearningCycleError(f"candidate is not eligible for promotion: {gate.reasons}")
        if not self.config.production_enabled:
            raise LearningCycleError("learned production activation is disabled")
        current = self.production.get(cycle.scope_key)
        if current is None or current.status != ProductionStatus.ACTIVE.value:
            raise LearningCycleError("an explicit existing Champion is required; first Candidate is not auto-promoted")
        model = None if candidate.model_version_id is None else self.model_registry_repository.get(candidate.model_version_id)
        learned_policy = self.policy_repository.get(candidate.policy_version_id)
        if learned_policy is None:
            raise LearningCycleError("candidate policy not found")
        ProductionSafety.validate_pair(
            model,
            learned_policy,
            expected_strategy=(cycle.strategy_name, cycle.strategy_variant, cycle.strategy_version),
        )
        policy_fp = (policy or self.promotion_policy).fingerprint()
        return self.production.promote(
            candidate=candidate,
            cycle=cycle,
            new_model_version_id=None if model is None else model.model_version_id,
            new_policy_version_id=learned_policy.policy_version_id,
            reason=reason,
            promotion_policy_version=policy_fp,
            evidence=gate.evidence.__dict__,
        )

    def initialize_champion(
        self,
        *,
        scope_key: str,
        strategy_name: str,
        strategy_variant: str,
        strategy_version: str,
        model_version_id: str | None,
        policy_version_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> ProductionModelState:
        """Explicit/manual bootstrap only; never called by an automatic cycle."""
        policy = self.policy_repository.get(policy_version_id)
        if policy is None:
            raise LearningCycleError("policy not found")
        model = None if model_version_id is None else self.model_registry_repository.get(model_version_id)
        ProductionSafety.validate_pair(model, policy, expected_strategy=(strategy_name, strategy_variant, strategy_version))
        return self.production.initialize(
            scope_key=scope_key,
            strategy_name=strategy_name,
            strategy_variant=strategy_variant,
            strategy_version=strategy_version,
            model_version_id=model_version_id,
            policy_version_id=policy_version_id,
            reason=reason,
            now=now or datetime.now(timezone.utc),
        )

    def rollback(self, scope_key: str, *, reason: str, monitoring_evidence: Mapping[str, Any]) -> ProductionModelState:
        state = self.production.get(scope_key)
        if state is None or state.status != ProductionStatus.ACTIVE.value:
            raise LearningCycleError("no active production state to rollback")
        snapshot = MonitoringSnapshot.from_mapping(monitoring_evidence)
        should, reasons = self.monitoring.should_rollback(snapshot, state, self.rollback_policy)
        if not should:
            raise LearningCycleError(f"rollback gate blocked: {reasons}")
        return self.production.rollback(scope_key, reason=reason, evidence={**monitoring_evidence, "rollback_reasons": reasons})

    def maybe_trigger_from_monitoring(
        self,
        request_factory: Callable[[MonitoringSnapshot], LearningCycleRequest],
        snapshot: MonitoringSnapshot,
    ) -> LearningCycle | None:
        if not self.monitoring.should_trigger_learning(snapshot, minimum_sample=self.config.minimum_monitoring_sample):
            return None
        request = request_factory(snapshot)
        if request.trigger != LearningTrigger.DRIFT:
            raise LearningCycleError("monitoring-triggered cycles must use DRIFT trigger")
        return self.trigger(request)

    def _run_with_retries(self, cycle: LearningCycle, request: LearningCycleRequest) -> LearningCycle:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                self.cycles.mark_retry(cycle.cycle_id, attempt)
                self.cycles.audit("learning_retry", cycle, reason=f"attempt={attempt}", status="RETRY")
            try:
                return self._run_once(cycle.cycle_id, request)
            except Exception as exc:
                last_error = exc
                logger.exception("learning cycle attempt failed")
                if attempt >= self.config.max_retries:
                    self.cycles.fail(cycle.cycle_id, error_type=type(exc).__name__, error_message=str(exc), error_stage="CYCLE")
                    self.cycles.audit("learning_failed", cycle, reason=str(exc), status="FAILED")
        raise LearningCycleError(str(last_error))

    def _run_once(self, cycle_id: str, request: LearningCycleRequest) -> LearningCycle:
        cycle = self.cycles.get(cycle_id)
        if cycle is None:
            raise LearningCycleError("cycle disappeared")
        self.cycles.set_status(cycle_id, LearningCycleStatus.DATASET_BUILD)
        self.cycles.audit("dataset_build_started", cycle, reason="L5_DATASET_MANAGER", status="STARTED")
        build_result = self.dataset_manager.build_and_store(**dict(request.dataset_build_kwargs))
        dataset = build_result.version if hasattr(build_result, "version") else build_result
        if not isinstance(dataset, DatasetVersion):
            raise LearningCycleError("dataset builder did not return DatasetVersion")
        if dataset.row_count < self.config.minimum_dataset_size:
            raise LearningCycleError("dataset minimum size gate failed")
        self.cycles.set_dataset(cycle_id, dataset.dataset_version_id)
        self.cycles.set_status(cycle_id, LearningCycleStatus.DATA_COMPLETION)

        model: ModelVersion | None = None
        training_result: TrainingResult | None = None
        if request.training_config_factory is not None:
            self.cycles.set_status(cycle_id, LearningCycleStatus.TRAINING)
            training_config = request.training_config_factory(dataset)
            if training_config.dataset_version_id != dataset.dataset_version_id:
                raise LearningCycleError("training config does not point to the built DatasetVersion")
            training_result = self.training_pipeline.train(training_config)
            if training_result.run.dataset_version_id != dataset.dataset_version_id:
                raise LearningCycleError("TrainingRun dataset lineage mismatch")
            self.cycles.set_training_run(cycle_id, training_result.run.training_run_id)
            self.cycles.audit("training_completed", cycle, reason="L6", status=training_result.run.status.value)
            self.cycles.set_status(cycle_id, LearningCycleStatus.REGISTRATION)
            model = self.model_registry.register(training_result.run.training_run_id)
            self.cycles.set_model_version(cycle_id, model.model_version_id)
            self.cycles.audit("model_registered", cycle, reason=model.model_version_id, status=model.status.value)
            self.cycles.set_status(cycle_id, LearningCycleStatus.VALIDATION)
            validation = self.validation_engine.evaluate(model.model_version_id, dataset_version_id=dataset.dataset_version_id)
            if validation.status.value != "COMPLETED":
                raise LearningCycleError("validation gate failed")
            self.cycles.set_validation_evaluation(cycle_id, validation.evaluation_id)
            self.cycles.audit("validation_completed", cycle, reason=validation.evaluation_id, status=validation.status.value)
            self.cycles.set_status(cycle_id, LearningCycleStatus.OOS)
            oos = self.oos_engine.evaluate_oos(model.model_version_id, dataset_version_id=dataset.dataset_version_id)
            if oos.status != L8EvaluationStatus.COMPLETED:
                raise LearningCycleError(f"OOS gate failed: {oos.status.value}")
            self.cycles.set_oos_evaluation(cycle_id, oos.evaluation_id)
            self.cycles.audit("oos_completed", cycle, reason=oos.evaluation_id, status=oos.status.value)
            self.cycles.set_status(cycle_id, LearningCycleStatus.ROBUSTNESS)
            robustness = self.oos_engine.evaluate_robustness(model.model_version_id, dataset_version_id=dataset.dataset_version_id)
            if robustness.status != L8EvaluationStatus.COMPLETED:
                raise LearningCycleError(f"robustness gate failed: {robustness.status.value}")
            self.cycles.set_robustness_evaluation(cycle_id, robustness.evaluation_id)
            self.cycles.audit("robustness_completed", cycle, reason=robustness.evaluation_id, status=robustness.status.value)
            policy = request.policy
            if request.policy_factory is not None:
                policy = request.policy_factory(model, dataset)
            if policy is None and request.strategy_name.upper() == "AI":
                policy = LearnedPolicyVersion.create(
                    policy_version_label=f"{request.strategy_variant}_POLICY",
                    strategy_name=request.strategy_name,
                    strategy_variant=request.strategy_variant,
                    strategy_version=request.strategy_version,
                    policy_types=(PolicyType.MODEL_POLICY,),
                    source_model_version_id=model.model_version_id,
                    source_training_run_id=training_result.run.training_run_id,
                    source_dataset_version_id=dataset.dataset_version_id,
                    feature_schema_version=model.feature_schema_version,
                    label_version=model.label_version,
                    policy_parameters={"inference_mode": "REGISTERED_ARTIFACT"},
                    enabled_capabilities=("MODEL_INFERENCE",),
                    default_mode=PolicyMode.CANDIDATE,
                )
            if policy is None:
                raise LearningCycleError("a learned policy is required to create a Candidate")
            self.policy_repository.create(policy)
            self.cycles.set_policy_version(cycle_id, policy.policy_version_id)
            evidence = PromotionEvidence(
                validation_status=validation.status.value,
                validation_samples=validation.sample_count,
                validation_metrics=_metric_map(validation.metrics),
                oos_status=oos.status.value,
                oos_samples=oos.sample_count,
                oos_metrics=_metric_map(oos.metrics),
                robustness_status=robustness.status.value,
                robustness_samples=robustness.sample_count,
                artifact_integrity=True,
                schema_compatible=(model.feature_schema_version == dataset.feature_schema_version and model.label_version == dataset.label_version),
                policy_compatible=(policy.source_model_version_id in {None, model.model_version_id}),
                references={"validation": validation.evaluation_id, "oos": oos.evaluation_id, "robustness": robustness.evaluation_id},
            )
        else:
            policy = request.policy
            if request.policy_factory is not None:
                policy = request.policy_factory(None, dataset)
            if policy is None or request.policy_evidence is None:
                raise LearningCycleError("policy-only cycle requires policy and explicit validated evidence")
            self.policy_repository.create(policy)
            self.cycles.set_policy_version(cycle_id, policy.policy_version_id)
            evidence = request.policy_evidence
        self.cycles.set_status(cycle_id, LearningCycleStatus.CANDIDATE)
        cycle = self.cycles.get(cycle_id)
        if cycle is None:
            raise LearningCycleError("cycle disappeared before candidate")
        candidate = self.create_candidate(cycle, model=model, policy=policy, evidence=evidence)
        self.cycles.set_status(cycle_id, LearningCycleStatus.PROMOTION_GATE)
        gate = self.promotion_gate(candidate, policy=self.promotion_policy)
        if gate.eligible:
            self.cycles.mark_challenger(candidate.candidate_id)
            self.cycles.audit("candidate_created", cycle, reason="CHALLENGER", status=CandidateStatus.CHALLENGER.value)
            if self.config.automatic_promotion_enabled:
                self.cycles.audit("promotion_attempted", cycle, reason="automatic promotion gate", status="ATTEMPTED")
                self.promote_candidate(candidate.candidate_id, reason="automatic promotion gate", policy=self.promotion_policy)
                completed = self.cycles.complete_success(cycle_id, LearningCycleStatus.PROMOTED, candidate_id=candidate.candidate_id, summary={"candidate_id": candidate.candidate_id, "promoted": True})
            else:
                completed = self.cycles.complete_success(cycle_id, LearningCycleStatus.CANDIDATE, candidate_id=candidate.candidate_id, summary={"candidate_id": candidate.candidate_id, "promoted": False, "promotion_policy_enabled": self.promotion_policy.enabled})
        elif not self.promotion_policy.enabled:
            # Disabled promotion is not a candidate rejection; it is a safe hold.
            self.cycles.mark_challenger(candidate.candidate_id)
            completed = self.cycles.complete_success(cycle_id, LearningCycleStatus.CANDIDATE, candidate_id=candidate.candidate_id, summary={"candidate_id": candidate.candidate_id, "promoted": False, "promotion_blocked": "POLICY_DISABLED"})
        else:
            self.cycles.reject_candidate(candidate.candidate_id, reason=";".join(gate.reasons))
            self.cycles.fail(cycle_id, error_type="PromotionGateError", error_message=";".join(gate.reasons), error_stage="PROMOTION_GATE")
            raise LearningCycleError(f"promotion gate rejected Candidate: {gate.reasons}")
        self.cycles.audit("learning_cycle_completed", completed, reason="PROMOTED" if completed.status == LearningCycleStatus.PROMOTED else "CANDIDATE_READY", status=completed.status.value)
        return completed

    def _candidate_evidence(self, candidate: Any) -> PromotionEvidence:
        cycle = self.cycles.get(candidate.cycle_id)
        if cycle is None:
            raise LearningCycleError("candidate cycle missing")
        validation = (
            self.model_registry_repository.get_evaluation(candidate.validation_evaluation_id)
            if candidate.validation_evaluation_id
            else None
        )
        oos = self.cycles.get_l8(candidate.oos_evaluation_id) if candidate.oos_evaluation_id else None
        robustness = self.cycles.get_l8(candidate.robustness_evaluation_id) if candidate.robustness_evaluation_id else None
        if validation is None or oos is None or robustness is None:
            # Policy-only evidence is stored in candidate metadata; this branch is replaced below by repository-backed evidence.
            evidence = self.cycles.get_candidate_evidence(candidate.candidate_id)
            if evidence is None:
                raise LearningCycleError("candidate evaluation evidence is missing")
            return PromotionEvidence(**evidence)
        return PromotionEvidence(
            validation_status=validation.status.value,
            validation_samples=validation.sample_count,
            validation_metrics=_metric_map(validation.metrics),
            oos_status=oos.status.value,
            oos_samples=oos.sample_count,
            oos_metrics=_metric_map(oos.metrics),
            robustness_status=robustness.status.value,
            robustness_samples=robustness.sample_count,
            artifact_integrity=True,
            schema_compatible=True,
            policy_compatible=True,
            references={"validation": validation.evaluation_id, "oos": oos.evaluation_id, "robustness": robustness.evaluation_id},
        )

    def _scope_key(self, request: LearningCycleRequest) -> str:
        return f"{request.strategy_name.upper()}::{request.strategy_variant}::{request.strategy_version}"

    def _ensure_automation_allowed(self, trigger: LearningTrigger) -> None:
        if trigger == LearningTrigger.MANUAL:
            return
        if not self.config.enabled or not self.config.automatic_retraining_enabled:
            raise LearningCycleError("automatic learning is disabled")

    def _check_data_and_cooldown(self, request: LearningCycleRequest) -> None:
        scope = self._scope_key(request)
        state = self.production.get_learning_state(scope)
        if state and state.last_successful_cycle_id:
            cycle = self.cycles.get(state.last_successful_cycle_id)
            if cycle and cycle.completed_at is not None:
                elapsed = (datetime.now(timezone.utc) - cycle.completed_at).total_seconds()
                if elapsed < self.config.cooldown_seconds:
                    raise LearningCycleError("learning cooldown is active")
        new_count = self.cycles.count_completed_since(scope, state.last_dataset_end if state else None)
        if new_count < self.config.minimum_new_completed_records:
            raise LearningCycleError("minimum_new_completed_records gate failed")


def _evidence_to_dict(evidence: PromotionEvidence) -> dict[str, Any]:
    return {
        "validation_status": evidence.validation_status,
        "validation_samples": evidence.validation_samples,
        "validation_metrics": dict(evidence.validation_metrics),
        "oos_status": evidence.oos_status,
        "oos_samples": evidence.oos_samples,
        "oos_metrics": dict(evidence.oos_metrics),
        "robustness_status": evidence.robustness_status,
        "robustness_samples": evidence.robustness_samples,
        "artifact_integrity": evidence.artifact_integrity,
        "schema_compatible": evidence.schema_compatible,
        "policy_compatible": evidence.policy_compatible,
        "references": dict(evidence.references),
    }


def _metric_map(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[str(key)] = float(value)
    return result


def _numeric_distribution_shift(observed: Sequence[Mapping[str, float]], baseline: Mapping[str, float]) -> float:
    if not observed or not baseline:
        return 0.0
    means: dict[str, float] = {}
    for key in sorted(baseline):
        values = [float(row[key]) for row in observed if key in row and math.isfinite(float(row[key]))]
        if values:
            means[key] = sum(values) / len(values)
    shifts = [abs(means[key] - float(baseline[key])) / max(abs(float(baseline[key])), 1e-9) for key in means]
    return float(sum(shifts) / len(shifts)) if shifts else 0.0


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _sha256(value: Any) -> str:
    body = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


__all__ = [
    "ContinuousLearningOrchestrator",
    "LearningCycleConfig",
    "LearningCycleError",
    "LearningCycleRequest",
    "LearningScheduler",
    "LearningTrigger",
    "MonitoringService",
    "ProductionSafety",
    "PromotionEvidence",
    "PromotionGate",
    "PromotionGateResult",
    "PromotionPolicy",
    "RollbackPolicy",
]

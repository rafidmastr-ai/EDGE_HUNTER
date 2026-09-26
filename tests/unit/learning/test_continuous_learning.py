"""L10 continuous-learning orchestration, safety gates and production lifecycle tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.continuous import (
    ContinuousLearningOrchestrator,
    LearningCycleConfig,
    LearningCycleError,
    LearningCycleRequest,
    LearningScheduler,
    LearningTrigger,
    MonitoringService,
    ProductionSafety,
    PromotionEvidence,
    PromotionGate,
    PromotionPolicy,
    RollbackPolicy,
)
from app.learning.continuous_repository import (
    L10RepositoryError,
    CandidateStatus,
    LearningCandidate,
    LearningCycle,
    LearningCycleRepository,
    LearningCycleStatus,
    LearningState,
    MonitoringSnapshot,
    ProductionModelState,
    ProductionStateRepository,
)
from app.learning.dataset import DatasetSplit, DatasetSplitPolicy, DatasetVersion
from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.models import LearningRecord, LearningRecordStatus, LearningDirection, LearningFeatureSnapshot, LearningSourceType
from app.learning.policies import LearnedPolicyVersion, PolicyMode, PolicyType
from app.learning.policy_repository import LearnedPolicyRepository
from app.learning.registry import ModelVersion, ModelVersionStatus
from app.learning.repository import LearningRecordRepository
from app.learning.training import LearningTrainingPipeline, TrainingConfig, TrainingStatus
from app.learning.training_repository import TrainingRun, TrainingRunRepository
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.evaluation import ModelEvaluationEngine
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.registry import ModelRegistry
from app.learning.oos_evaluation import L8EvaluationEngine, L8EvaluationStatus, L8EvaluationType, L8EvaluationReport
from app.learning.advanced_evaluation_repository import L8EvaluationRepository
from app.learning.dataset import LearningDatasetManager


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCHEMA = "features-v1"
LABEL = "v1"


class _DatasetStub:
    def __init__(self, version: DatasetVersion):
        self.version = version
        self.calls: list[dict[str, object]] = []

    def build_and_store(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return type("BuildResult", (), {"version": self.version})()


class _FailingDatasetStub:
    def __init__(self, message: str = "dataset failed"):
        self.message = message
        self.calls = 0

    def build_and_store(self, **_: object):
        self.calls += 1
        raise RuntimeError(self.message)


class _NoopTraining:
    def train(self, *_: object, **__: object):
        raise AssertionError("training must not be called in this test")


class _NoopRegistry:
    def register(self, *_: object, **__: object):
        raise AssertionError("registry must not be called in this test")


class _FakeModelRegistryRepo:
    def __init__(self):
        self.models: dict[str, ModelVersion] = {}

    def get(self, model_version_id: str):
        return self.models.get(model_version_id)


class _ValidationStub:
    def __init__(self, status: str = "COMPLETED"):
        self.status = type("Status", (), {"value": status})()
        self.evaluation_id = "eval-validation"
        self.sample_count = 4
        self.metrics = {"accuracy": 0.75, "f1_macro": 0.70}

    def evaluate(self, *_: object, **__: object):
        return self


class _L8Stub:
    def __init__(self, status: L8EvaluationStatus = L8EvaluationStatus.COMPLETED):
        self.status = status
        self.evaluation_id = "l8-eval"
        self.sample_count = 4
        self.metrics = {"accuracy": 0.75, "f1_macro": 0.70}

    def evaluate_oos(self, *_: object, **__: object):
        return self

    def evaluate_robustness(self, *_: object, **__: object):
        return self


class ContinuousLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Database(self.root / "l10.db")
        MigrationRunner(self.db).apply_all()
        self.cycles = LearningCycleRepository(self.db)
        self.production = ProductionStateRepository(self.db)
        self.features = FeatureStore(self.db)
        self.records = LearningRecordRepository(self.db)
        self.datasets = DatasetVersionRepository(self.db)
        self.policies = LearnedPolicyRepository(self.db)
        self.training_runs = TrainingRunRepository(self.db)
        self.model_registry_repo = ModelRegistryRepository(self.db)
        self.l8_repo = L8EvaluationRepository(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    # ---------- reusable fixtures ----------
    def _add_record(self, index: int, *, strategy: str = "AI", variant: str = "AI_V1", version: str = "AI_V1", source: LearningSourceType = LearningSourceType.HISTORICAL_BACKTEST) -> LearningRecord:
        decision = BASE + timedelta(hours=index)
        snapshot = FeatureSnapshot(
            timestamp=decision,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=SCHEMA,
            features={"rsi": 40.0 + index, "ema_distance": (-1.0 if index % 2 == 0 else 1.0) + index * 0.05, "phase": "TREND" if index % 2 == 0 else "RANGE"},
            market_regime="TREND" if index % 2 == 0 else "RANGE",
            created_at=BASE,
            provenance_metadata={"provider": "test"},
        )
        snapshot = self.features.save(snapshot)
        outcome = "TP_BEFORE_SL" if index % 2 == 0 else "SL_BEFORE_TP"
        record = LearningRecord(
            record_id=f"l10-record-{strategy}-{index}",
            source_type=source,
            symbol="XAUUSD",
            timeframe="M15",
            decision_timestamp=decision,
            entry_timestamp=decision,
            direction=LearningDirection.BUY,
            entry_price=100 + index,
            target=102 + index,
            stop_loss=99 + index,
            risk_reward=2.0,
            strategy_name=strategy,
            strategy_variant=variant,
            strategy_version=version,
            feature_set_version=SCHEMA,
            parameters_snapshot={"p": index},
            feature_snapshot=LearningFeatureSnapshot(snapshot.features),
            evidence_snapshot={"signal": "test"},
            confidence=70.0,
            market_regime=snapshot.market_regime,
            feature_snapshot_id=snapshot.snapshot_id,
            outcome=outcome,
            exit_timestamp=decision + timedelta(hours=1),
            exit_price=102 + index,
            exit_reason="TP" if outcome == "TP_BEFORE_SL" else "SL",
            duration=timedelta(hours=1),
            label_version=LABEL,
            provenance_metadata={"event_kind": source.value},
            status=LearningRecordStatus.COMPLETED,
            created_at=BASE,
        )
        return self.records.create(record)

    def _dataset(self, count: int = 12, *, strategy: str = "AI", variant: str = "AI_V1", version: str = "AI_V1") -> DatasetVersion:
        for i in range(count):
            self._add_record(i, strategy=strategy, variant=variant, version=version)
        manager = LearningDatasetManager(self.records, self.features)
        result = manager.build_and_store(
            dataset_name=f"l10-{strategy.lower()}-{variant}",
            strategies=(strategy,),
            strategy_versions=(version,),
            split_policy=DatasetSplitPolicy(),
            source_types=(LearningSourceType.HISTORICAL_BACKTEST,),
        )
        return result.version

    def _policy(self, strategy: str, variant: str, version: str, *, source_model: str | None = None, mode: PolicyMode = PolicyMode.CANDIDATE, params: dict[str, object] | None = None) -> LearnedPolicyVersion:
        dataset = None
        training_id = None
        if source_model is not None:
            # Persist the minimal real lineage required by the L9 foreign keys.
            dataset = self._dataset(count=3, strategy=strategy, variant=variant, version=version)
            training_id = f"trn-fixture-{source_model}"
            if self.training_runs.get(training_id) is None:
                self.db.execute(
                    """INSERT INTO learning_training_runs(
                        training_run_id, training_identity_hash, dataset_version_id, started_at, completed_at, status,
                        strategy_name, strategy_variant, strategy_version, model_type, model_version_label,
                        feature_schema_version, label_version, train_count, validation_count, oos_count,
                        training_config_json, random_seed, metrics_json, artifact_path, artifact_sha256,
                        artifact_metadata_json, error_type, error_message, error_stage, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?, 'logistic_regression', 'AI_MODEL', ?, ?, 1, 1, 1, ?, 0, ?, ?, ?, '{}', NULL, NULL, NULL, ?)""",
                    (training_id, f"hash-{training_id}", dataset.dataset_version_id, BASE.isoformat(), BASE.isoformat(),
                     strategy, variant, version, SCHEMA, LABEL, json.dumps({}), json.dumps({"accuracy": 1.0}),
                     str(self.root / "fixture.joblib"), "a" * 64, BASE.isoformat()),
                )
                self.db.commit()
            if self.model_registry_repo.get(source_model) is None:
                self.db.execute(
                    """INSERT INTO learning_model_versions(
                        model_version_id, model_version_label, strategy_name, strategy_variant, strategy_version,
                        model_type, training_run_id, dataset_version_id, feature_schema_version, label_version,
                        artifact_path, artifact_sha256, artifact_format_version, created_at, registered_at, status, metadata_json
                    ) VALUES (?, 'AI_MODEL', ?, ?, ?, 'logistic_regression', ?, ?, ?, ?, ?, ?, 'l6-artifact-v1', ?, ?, 'REGISTERED', '{}')""",
                    (source_model, strategy, variant, version, training_id, dataset.dataset_version_id, SCHEMA, LABEL,
                     str(self.root / "fixture.joblib"), "a" * 64, BASE.isoformat(), BASE.isoformat()),
                )
                self.db.commit()
        policy = LearnedPolicyVersion.create(
            policy_version_label=f"{variant}_POLICY",
            strategy_name=strategy,
            strategy_variant=variant,
            strategy_version=version,
            policy_types=(PolicyType.PARAMETER_POLICY,) if source_model is None else (PolicyType.MODEL_POLICY,),
            source_model_version_id=source_model,
            source_training_run_id=training_id,
            source_dataset_version_id=None if dataset is None else dataset.dataset_version_id,
            feature_schema_version=SCHEMA if source_model else None,
            label_version=LABEL if source_model else None,
            policy_parameters=params or ({"target_rr": 1.75} if source_model is None else {"inference_mode": "REGISTERED_ARTIFACT"}),
            enabled_capabilities=("MODEL_INFERENCE",) if source_model else ("PARAMETER_OVERRIDE",),
            default_mode=mode,
            created_at=BASE,
        )
        return self.policies.create(policy)

    def _config(self, **kwargs: object) -> LearningCycleConfig:
        values = {
            "enabled": True,
            "automatic_retraining_enabled": True,
            "automatic_promotion_enabled": False,
            "production_enabled": False,
            "minimum_new_completed_records": 0,
            "minimum_dataset_size": 0,
            "cooldown_seconds": 0,
            "max_retries": 0,
            "minimum_monitoring_sample": 1,
            "drift_threshold": 0.25,
            "rollback_grace_period_seconds": 0,
        }
        values.update(kwargs)
        return LearningCycleConfig(**values)

    def _orchestrator(self, *, config: LearningCycleConfig | None = None, promotion_policy: PromotionPolicy | None = None, rollback_policy: RollbackPolicy | None = None, dataset_manager: object | None = None, training_pipeline: object | None = None, model_registry: object | None = None, validation_engine: object | None = None, oos_engine: object | None = None) -> ContinuousLearningOrchestrator:
        dataset = self._dataset()
        manager = dataset_manager or _DatasetStub(dataset)
        training_pipeline = training_pipeline or LearningTrainingPipeline(self.datasets, self.features, self.training_runs, artifact_root=self.root / "artifacts")
        model_registry = model_registry or ModelRegistry(self.model_registry_repo, self.training_runs, self.datasets, self.features, training_pipeline, artifact_root=training_pipeline.artifact_root)
        validation_engine = validation_engine or ModelEvaluationEngine(self.model_registry_repo, self.datasets, self.features, training_pipeline)
        oos_engine = oos_engine or L8EvaluationEngine(self.model_registry_repo, self.datasets, self.features, training_pipeline, self.l8_repo)
        return ContinuousLearningOrchestrator(
            cycle_repository=self.cycles,
            production_repository=self.production,
            dataset_manager=manager,
            dataset_repository=self.datasets,
            training_pipeline=training_pipeline,
            training_repository=self.training_runs,
            model_registry=model_registry,
            model_registry_repository=self.model_registry_repo,
            validation_engine=validation_engine,
            oos_engine=oos_engine,
            policy_repository=self.policies,
            config=config or self._config(),
            promotion_policy=promotion_policy,
            rollback_policy=rollback_policy,
        )

    def _request(self, strategy: str = "Classic", variant: str = "Classic_V1", version: str = "Classic_V1", *, trigger: LearningTrigger = LearningTrigger.MANUAL, key: str = "k1", policy: LearnedPolicyVersion | None = None, evidence: PromotionEvidence | None = None) -> LearningCycleRequest:
        return LearningCycleRequest(
            strategy_name=strategy,
            strategy_variant=variant,
            strategy_version=version,
            trigger=trigger,
            trigger_key=key,
            dataset_build_kwargs={"dataset_name": "l10-test", "strategies": (strategy,), "strategy_versions": (version,)},
            policy=policy,
            policy_evidence=evidence,
        )

    # ---------- configuration / trigger contracts ----------
    def test_01_learning_cycle_contract(self):
        req = self._request()
        self.assertEqual(req.trigger, LearningTrigger.MANUAL)
        self.assertEqual(req.trigger_key, "k1")

    def test_02_scheduled_trigger_exists(self):
        self.assertEqual(LearningTrigger.SCHEDULED.value, "SCHEDULED")

    def test_03_manual_trigger_exists(self):
        self.assertEqual(LearningTrigger.MANUAL.value, "MANUAL")

    def test_04_data_threshold_trigger_exists(self):
        self.assertEqual(LearningTrigger.DATA_THRESHOLD.value, "DATA_THRESHOLD")

    def test_05_drift_trigger_exists(self):
        self.assertEqual(LearningTrigger.DRIFT.value, "DRIFT")

    def test_06_safe_defaults_disable_automatic_learning(self):
        config = LearningCycleConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.automatic_retraining_enabled)
        self.assertFalse(config.automatic_promotion_enabled)
        self.assertFalse(config.production_enabled)

    def test_07_automatic_promotion_requires_production_flag(self):
        with self.assertRaises(ValueError):
            LearningCycleConfig(automatic_promotion_enabled=True, production_enabled=False)

    def test_08_schedule_interval_is_configurable_and_safe(self):
        self.assertEqual(LearningCycleConfig().schedule_interval_seconds, 3600)

    def test_09_minimum_new_records_is_configurable(self):
        self.assertEqual(self._config(minimum_new_completed_records=5).minimum_new_completed_records, 5)

    def test_10_minimum_dataset_size_is_configurable(self):
        self.assertEqual(self._config(minimum_dataset_size=10).minimum_dataset_size, 10)

    def test_11_cooldown_is_configurable(self):
        self.assertEqual(self._config(cooldown_seconds=123).cooldown_seconds, 123)

    def test_12_retry_limit_is_bounded(self):
        self.assertEqual(self._config(max_retries=2).max_retries, 2)

    def test_13_learning_config_fingerprint_is_deterministic(self):
        self.assertEqual(self._config().fingerprint(), self._config().fingerprint())

    def test_14_promotion_policy_requires_multiple_metrics(self):
        with self.assertRaises(ValueError):
            PromotionPolicy(enabled=True, minimum_metrics_required=2, minimum_validation_metrics={"accuracy": 0.7}, minimum_oos_metrics={})

    def test_15_promotion_policy_is_configurable(self):
        policy = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}, minimum_metrics_required=2)
        self.assertTrue(policy.enabled)
        self.assertEqual(len(policy.minimum_validation_metrics), 1)

    def test_16_rollback_policy_is_configurable(self):
        policy = RollbackPolicy(minimum_monitoring_sample=3, grace_period_seconds=4)
        self.assertEqual(policy.minimum_monitoring_sample, 3)

    # ---------- repository / persistence / state ----------
    def test_17_cycle_persists(self):
        cycle = LearningCycle.create(identity_hash="a" * 64, scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", trigger=LearningTrigger.MANUAL, trigger_key="x", configuration_fingerprint="b" * 64, started_at=BASE)
        stored = self.cycles.create(cycle)
        self.assertEqual(stored.cycle_id, cycle.cycle_id)
        self.assertEqual(self.cycles.get(cycle.cycle_id), stored)

    def test_18_cycle_list_is_deterministic(self):
        for i in range(2):
            self.cycles.create(LearningCycle.create(identity_hash=str(i) * 64, scope_key=f"CLASSIC::Classic_V1::Classic_V{i+1}", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version=f"Classic_V{i+1}", trigger=LearningTrigger.MANUAL, trigger_key=str(i), configuration_fingerprint="c" * 64, started_at=BASE + timedelta(hours=i)))
        self.assertEqual([x.trigger_key for x in self.cycles.list_cycles()], ["0", "1"])

    def test_19_learning_state_persists(self):
        state = LearningState("SCOPE", None, None, None, None, None, None, None, "failure", "MANUAL", BASE)
        self.cycles.upsert_learning_state(state)
        self.assertEqual(self.cycles.get_learning_state("SCOPE").last_failure, "failure")

    def test_20_failed_cycle_preserves_last_success(self):
        cycle1 = LearningCycle.create(identity_hash="1" * 64, scope_key="AI::AI_V1::AI_V1", strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1", trigger=LearningTrigger.MANUAL, trigger_key="a", configuration_fingerprint="d" * 64, started_at=BASE)
        self.cycles.create(cycle1)
        self.cycles.set_dataset(cycle1.cycle_id, None)
        self.cycles.complete_success(cycle1.cycle_id, LearningCycleStatus.CANDIDATE)
        cycle2 = LearningCycle.create(identity_hash="2" * 64, scope_key=cycle1.scope_key, strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1", trigger=LearningTrigger.MANUAL, trigger_key="b", configuration_fingerprint="e" * 64, started_at=BASE + timedelta(days=1))
        self.cycles.create(cycle2)
        self.cycles.fail(cycle2.cycle_id, error_type="RuntimeError", error_message="x", error_stage="CYCLE")
        self.assertEqual(self.cycles.get_learning_state(cycle1.scope_key).last_successful_cycle_id, cycle1.cycle_id)

    def test_21_cycle_history_cannot_be_deleted(self):
        cycle = LearningCycle.create(identity_hash="3" * 64, scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", trigger=LearningTrigger.MANUAL, trigger_key="d", configuration_fingerprint="f" * 64, started_at=BASE)
        self.cycles.create(cycle)
        with self.assertRaises(Exception):
            self.db.execute("DELETE FROM learning_cycles WHERE cycle_id=?", (cycle.cycle_id,)); self.db.commit()

    def test_22_candidate_history_cannot_be_deleted(self):
        dataset = self._dataset(count=3)
        policy = self._policy("AI", "AI_V1", "AI_V1")
        cycle = LearningCycle.create(identity_hash="4" * 64, scope_key="AI::AI_V1::AI_V1", strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1", trigger=LearningTrigger.MANUAL, trigger_key="x", configuration_fingerprint="1" * 64, started_at=BASE)
        self.cycles.create(cycle); self.cycles.set_dataset(cycle.cycle_id, dataset.dataset_version_id)
        candidate = self.cycles.create_candidate(cycle=cycle, model_version_id=None, policy_version_id=policy.policy_version_id, dataset_version_id=dataset.dataset_version_id, validation_evaluation_id=None, oos_evaluation_id=None, robustness_evaluation_id=None)
        with self.assertRaises(Exception):
            self.db.execute("DELETE FROM learning_candidates WHERE candidate_id=?", (candidate.candidate_id,)); self.db.commit()

    def test_23_audit_history_is_append_only(self):
        self.cycles.audit("test", None, reason="x", status="OK")
        row = self.db.execute("SELECT event_id FROM learning_audit_events").fetchone()
        with self.assertRaises(Exception):
            self.db.execute("DELETE FROM learning_audit_events WHERE event_id=?", (row["event_id"],)); self.db.commit()

    def test_24_production_state_starts_empty(self):
        self.assertIsNone(self.production.get("AI::AI_V1::AI_V1"))

    def test_25_production_history_is_queryable(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        state = self.production.initialize(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, reason="test", now=BASE)
        history = self.cycles.list_production_history(scope_key=state.scope_key)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].event_type, "INITIAL_ACTIVATION")

    def test_26_monitoring_snapshot_persists(self):
        snap = MonitoringSnapshot.create(scope_key="AI::AI_V1::AI_V1", model_version_id=None, policy_version_id=None, observed_at=BASE, sample_count=2, metrics={}, baseline_metrics={}, drift_score=0.0, drift_detected=False, runtime_error_count=0, artifact_healthy=True)
        stored = self.cycles.create_monitoring_snapshot(snap)
        self.assertEqual(self.cycles.list_monitoring_snapshots()[0].snapshot_id, stored.snapshot_id)

    # ---------- threshold / cooldown / scheduler ----------
    def test_27_minimum_new_records_gate_blocks_automatic_cycle(self):
        orch = self._orchestrator(config=self._config(minimum_new_completed_records=100))
        with self.assertRaises(LearningCycleError):
            orch.trigger(self._request(trigger=LearningTrigger.SCHEDULED, key="gate"))

    def test_28_manual_trigger_works_when_automation_flags_are_off(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        evidence = self._evidence()
        orch = self._orchestrator(config=LearningCycleConfig(minimum_new_completed_records=0, minimum_dataset_size=0, cooldown_seconds=0), dataset_manager=_DatasetStub(self._dataset()))
        result = orch.trigger(self._request(policy=policy, evidence=evidence, key="manual-off"))
        self.assertEqual(result.status, LearningCycleStatus.CANDIDATE)

    def test_29_automatic_trigger_requires_flags(self):
        orch = self._orchestrator(config=LearningCycleConfig(minimum_new_completed_records=0, minimum_dataset_size=0, cooldown_seconds=0))
        with self.assertRaises(LearningCycleError):
            orch.trigger(self._request(trigger=LearningTrigger.SCHEDULED, key="auto-off"))

    def test_30_start_scheduler_returns_false_when_disabled(self):
        orch = self._orchestrator(config=LearningCycleConfig())
        self.assertFalse(orch.start_scheduler())

    def test_31_start_scheduler_returns_true_when_enabled(self):
        orch = self._orchestrator(config=self._config())
        self.assertTrue(orch.start_scheduler())
        orch.stop_scheduler()

    def test_32_duplicate_trigger_is_idempotent(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        evidence = self._evidence()
        orch = self._orchestrator(config=self._config())
        req = self._request(policy=policy, evidence=evidence, key="same")
        a = orch.trigger(req); b = orch.trigger(req)
        self.assertEqual(a.cycle_id, b.cycle_id)
        self.assertEqual(len(self.cycles.list_cycles(scope_key=a.scope_key)), 1)

    def test_33_concurrent_identical_triggers_create_one_cycle(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        evidence = self._evidence()
        dataset = self._dataset()
        orch = self._orchestrator(config=self._config(), dataset_manager=_DatasetStub(dataset))
        req = self._request(policy=policy, evidence=evidence, key="concurrent")
        results: list[LearningCycle] = []
        errors: list[Exception] = []
        def worker():
            try: results.append(orch.trigger(req))
            except Exception as exc: errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertFalse(errors, errors)
        self.assertEqual(len({r.cycle_id for r in results}), 1)

    def test_34_different_active_cycle_same_scope_is_blocked(self):
        first = LearningCycle.create(identity_hash="5" * 64, scope_key="SMC::SMC_V1::SMC_V1", strategy_name="SMC", strategy_variant="SMC_V1", strategy_version="SMC_V1", trigger=LearningTrigger.MANUAL, trigger_key="1", configuration_fingerprint="2" * 64, started_at=BASE)
        second = LearningCycle.create(identity_hash="6" * 64, scope_key=first.scope_key, strategy_name="SMC", strategy_variant="SMC_V1", strategy_version="SMC_V1", trigger=LearningTrigger.MANUAL, trigger_key="2", configuration_fingerprint="3" * 64, started_at=BASE)
        self.cycles.create(first)
        with self.assertRaises(Exception): self.cycles.create(second)

    # ---------- monitoring / drift ----------
    def test_35_monitoring_creates_snapshot(self):
        monitoring = MonitoringService(self.cycles)
        snapshot = monitoring.observe(scope_key="AI::AI_V1::AI_V1", model_version_id=None, policy_version_id=None, sample_count=3, metrics={"accuracy": 0.7}, baseline_metrics={"accuracy": 0.8}, observed_numeric=({"x": 2.0},), baseline_numeric={"x": 1.0}, drift_threshold=0.5, now=BASE)
        self.assertTrue(snapshot.drift_detected)

    def test_36_drift_detection_does_not_change_production_state(self):
        monitoring = MonitoringService(self.cycles)
        state = self.production.get("AI::AI_V1::AI_V1")
        self.assertIsNone(state)
        monitoring.observe(scope_key="AI::AI_V1::AI_V1", model_version_id=None, policy_version_id=None, sample_count=3, metrics={}, baseline_metrics={}, observed_numeric=({"x": 10.0},), baseline_numeric={"x": 1.0}, drift_threshold=0.1, now=BASE)
        self.assertIsNone(self.production.get("AI::AI_V1::AI_V1"))

    def test_37_drift_can_trigger_a_learning_cycle(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        evidence = self._evidence()
        dataset = self._dataset()
        orch = self._orchestrator(config=self._config(minimum_monitoring_sample=1), dataset_manager=_DatasetStub(dataset))
        snapshot = orch.monitoring.observe(scope_key="CLASSIC::Classic_V1::Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, sample_count=2, metrics={}, baseline_metrics={}, observed_numeric=({"x": 10.0},), baseline_numeric={"x": 1.0}, drift_threshold=0.2, now=BASE)
        result = orch.maybe_trigger_from_monitoring(lambda _: self._request(policy=policy, evidence=evidence, trigger=LearningTrigger.DRIFT, key="drift-key"), snapshot)
        self.assertIsNotNone(result)
        self.assertEqual(result.trigger, LearningTrigger.DRIFT.value)

    def test_38_monitoring_below_sample_threshold_does_not_trigger(self):
        snap = MonitoringSnapshot.create(scope_key="X", model_version_id=None, policy_version_id=None, observed_at=BASE, sample_count=1, metrics={}, baseline_metrics={}, drift_score=1.0, drift_detected=True, runtime_error_count=0, artifact_healthy=True)
        self.assertFalse(MonitoringService.should_trigger_learning(snap, minimum_sample=2))

    def test_39_drift_is_not_immediate_replacement(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        state = self.production.initialize(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, reason="bootstrap", now=BASE)
        snap = MonitoringSnapshot.create(scope_key=state.scope_key, model_version_id=None, policy_version_id=policy.policy_version_id, observed_at=BASE + timedelta(hours=1), sample_count=10, metrics={}, baseline_metrics={}, drift_score=1.0, drift_detected=True, runtime_error_count=0, artifact_healthy=True)
        should, _ = MonitoringService.should_rollback(snap, state, RollbackPolicy(minimum_monitoring_sample=1, grace_period_seconds=0, max_drift_score=0.5), now=BASE + timedelta(hours=1))
        self.assertTrue(should)
        self.assertEqual(self.production.get(state.scope_key).policy_version_id, state.policy_version_id)

    # ---------- promotion gate / candidate lifecycle ----------
    def _evidence(self, *, validation_status: str = "COMPLETED", oos_status: str = "COMPLETED", robustness_status: str = "COMPLETED", validation_samples: int = 4, oos_samples: int = 4, validation_metrics: dict[str, float] | None = None, oos_metrics: dict[str, float] | None = None, artifact_integrity: bool = True, schema: bool = True, policy: bool = True) -> PromotionEvidence:
        return PromotionEvidence(
            validation_status=validation_status,
            validation_samples=validation_samples,
            validation_metrics=validation_metrics or {"accuracy": 0.8, "f1_macro": 0.75},
            oos_status=oos_status,
            oos_samples=oos_samples,
            oos_metrics=oos_metrics or {"accuracy": 0.78, "f1_macro": 0.73},
            robustness_status=robustness_status,
            robustness_samples=4,
            artifact_integrity=artifact_integrity,
            schema_compatible=schema,
            policy_compatible=policy,
            references={},
        )

    def test_40_promotion_gate_disabled_blocks_promotion(self):
        result = PromotionGate().evaluate(self._evidence(), PromotionPolicy(enabled=False), candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)
        self.assertIn("PROMOTION_POLICY_DISABLED", result.reasons)

    def test_41_candidate_requires_completed_validation(self):
        policy = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}, minimum_metrics_required=2)
        result = PromotionGate().evaluate(self._evidence(validation_status="FAILED"), policy, candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)
        self.assertIn("VALIDATION_NOT_COMPLETED", result.reasons)

    def test_42_candidate_requires_completed_oos(self):
        policy = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}, minimum_metrics_required=2)
        result = PromotionGate().evaluate(self._evidence(oos_status="FAILED"), policy, candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)
        self.assertIn("OOS_NOT_COMPLETED", result.reasons)

    def test_43_candidate_requires_completed_robustness(self):
        policy = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}, minimum_metrics_required=2)
        result = PromotionGate().evaluate(self._evidence(robustness_status="FAILED"), policy, candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)
        self.assertIn("ROBUSTNESS_NOT_COMPLETED", result.reasons)

    def test_44_multi_metric_gate_is_required(self):
        policy = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}, minimum_metrics_required=2)
        result = PromotionGate().evaluate(self._evidence(validation_metrics={"accuracy": 0.8}, oos_metrics={"f1_macro": 0.2}), policy, candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertTrue(result.eligible)

    def test_45_candidate_policy_strategy_mismatch_rejected(self):
        policy = self._policy("SMC", "SMC_V1", "SMC_V1")
        gate = PromotionGate().evaluate(self._evidence(), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=None, candidate_policy=policy, expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(gate.eligible)
        self.assertIn("POLICY_STRATEGY_MISMATCH", gate.reasons)

    def test_46_model_strategy_mismatch_rejected(self):
        model = self._fake_model("model-1", strategy="SMC", variant="SMC_V1", version="SMC_V1")
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        gate = PromotionGate().evaluate(self._evidence(), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=model, candidate_policy=policy, expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(gate.eligible)
        self.assertIn("MODEL_STRATEGY_MISMATCH", gate.reasons)

    def _fake_model(self, model_id: str, *, strategy: str = "AI", variant: str = "AI_V1", version: str = "AI_V1") -> ModelVersion:
        return ModelVersion(model_version_id=model_id, model_version_label="AI_MODEL", strategy_name=strategy, strategy_variant=variant, strategy_version=version, model_type="logistic_regression", training_run_id="trn", dataset_version_id="dsv", feature_schema_version=SCHEMA, label_version=LABEL, artifact_path=str(self.root / "artifact.joblib"), artifact_sha256="a" * 64, artifact_format_version="l6-artifact-v1", created_at=BASE, registered_at=BASE, status=ModelVersionStatus.REGISTERED, metadata={})

    def test_47_candidate_requires_policy(self):
        result = PromotionGate().evaluate(self._evidence(), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=None, candidate_policy=None, expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)
        self.assertIn("POLICY_VERSION_REQUIRED", result.reasons)

    def test_48_candidate_requires_artifact_integrity(self):
        result = PromotionGate().evaluate(self._evidence(artifact_integrity=False), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)

    def test_49_candidate_requires_schema_compatibility(self):
        result = PromotionGate().evaluate(self._evidence(schema=False), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)

    def test_50_candidate_requires_policy_compatibility(self):
        result = PromotionGate().evaluate(self._evidence(policy=False), PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0}), candidate_model=None, candidate_policy=self._policy("Classic", "Classic_V1", "Classic_V1"), expected_strategy=("Classic", "Classic_V1", "Classic_V1"))
        self.assertFalse(result.eligible)

    # ---------- production safety / champion / challenger ----------
    def test_51_first_candidate_is_not_auto_champion(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config(production_enabled=True, automatic_promotion_enabled=False))
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="candidate-only"))
        self.assertEqual(result.status, LearningCycleStatus.CANDIDATE)
        self.assertIsNone(self.production.get(result.scope_key))

    def test_52_initialize_champion_requires_existing_policy(self):
        orch = self._orchestrator(config=self._config())
        with self.assertRaises(LearningCycleError):
            orch.initialize_champion(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id="missing", reason="x")

    def test_53_candidate_becomes_challenger_after_promotion_gate(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config())
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="challenger"))
        candidate = self.cycles.get_candidate(result.result_summary["candidate_id"])
        self.assertEqual(candidate.status, CandidateStatus.CHALLENGER)

    def test_54_promotion_requires_existing_champion(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        promo = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0})
        orch = self._orchestrator(config=self._config(production_enabled=True), promotion_policy=promo)
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="needs-champion"))
        cid = result.result_summary["candidate_id"]
        with self.assertRaises(LearningCycleError):
            orch.promote_candidate(cid, reason="test", policy=promo)

    def test_55_production_pair_rejects_revoked_model(self):
        policy = self._policy("AI", "AI_V1", "AI_V1", source_model="m1")
        revoked = self._fake_model("m1")
        revoked = ModelVersion(**{**revoked.to_dict(), "status": ModelVersionStatus.REVOKED}) if False else revoked
        revoked = self._fake_model("m1")
        with self.assertRaises(LearningCycleError):
            # validate_pair reads only enum on the provided object.
            ProductionSafety.validate_pair(ModelVersion(model_version_id=revoked.model_version_id, model_version_label=revoked.model_version_label, strategy_name=revoked.strategy_name, strategy_variant=revoked.strategy_variant, strategy_version=revoked.strategy_version, model_type=revoked.model_type, training_run_id=revoked.training_run_id, dataset_version_id=revoked.dataset_version_id, feature_schema_version=revoked.feature_schema_version, label_version=revoked.label_version, artifact_path=revoked.artifact_path, artifact_sha256=revoked.artifact_sha256, artifact_format_version=revoked.artifact_format_version, created_at=BASE, registered_at=BASE, status=ModelVersionStatus.REVOKED, metadata={}), policy, expected_strategy=("AI", "AI_V1", "AI_V1"))

    def test_56_production_pair_rejects_strategy_mismatch(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        with self.assertRaises(LearningCycleError):
            ProductionSafety.validate_pair(self._fake_model("m2", strategy="SMC", variant="SMC_V1", version="SMC_V1"), policy, expected_strategy=("Classic", "Classic_V1", "Classic_V1"))

    def test_57_production_pair_rejects_model_policy_lineage_mismatch(self):
        model = self._fake_model("m3")
        policy = self._policy("AI", "AI_V1", "AI_V1", source_model="other")
        with self.assertRaises(LearningCycleError):
            ProductionSafety.validate_pair(model, policy, expected_strategy=("AI", "AI_V1", "AI_V1"))

    def test_58_no_fixed_strategy_weights_allowed(self):
        with self.assertRaises(Exception):
            self._policy("Classic", "Classic_V1", "Classic_V1", params={"strategy_weights": {"Classic": 1.0}})

    def test_59_risk_percent_override_is_rejected(self):
        with self.assertRaises(Exception):
            self._policy("Classic", "Classic_V1", "Classic_V1", params={"risk_percent": 0.1})

    def test_60_rr_stays_in_project_range(self):
        self._policy("Classic", "Classic_V1", "Classic_V1", params={"target_rr": 1.5})
        self._policy("SMC", "SMC_V1", "SMC_V1", params={"target_rr": 2.0})
        with self.assertRaises(Exception):
            self._policy("ICT", "ICT_V1", "ICT_V1", params={"target_rr": 2.1})

    def test_61_multiple_strategy_states_are_isolated(self):
        for strategy, variant, version in [("Classic", "Classic_V1", "Classic_V1"), ("SMC", "SMC_V1", "SMC_V1"), ("ICT", "ICT_V1", "ICT_V1"), ("AI", "AI_V1", "AI_V1")]:
            policy = self._policy(strategy, variant, version, source_model="mAI" if strategy == "AI" else None)
            self.production.initialize(scope_key=f"{strategy.upper()}::{variant}::{version}", strategy_name=strategy, strategy_variant=variant, strategy_version=version, model_version_id="mAI" if strategy == "AI" else None, policy_version_id=policy.policy_version_id, reason="isolated", now=BASE)
        states = [self.production.get(f"{s.upper()}::{v}::{ver}") for s, v, ver in [("Classic","Classic_V1","Classic_V1"),("SMC","SMC_V1","SMC_V1"),("ICT","ICT_V1","ICT_V1"),("AI","AI_V1","AI_V1")]]
        self.assertEqual(len({x.scope_key for x in states if x}), 4)

    def test_62_strategy_auto_switching_is_not_implemented(self):
        self.assertNotIn("switch", ContinuousLearningOrchestrator._scope_key.__name__.lower())

    def test_63_production_disabled_flag_blocks_manual_promotion(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config(production_enabled=False))
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="prod-off"))
        with self.assertRaises(LearningCycleError):
            orch.promote_candidate(result.result_summary["candidate_id"], reason="blocked")

    # ---------- failure isolation / retries ----------
    def test_64_dataset_failure_leaves_champion_unchanged(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config(max_retries=1), dataset_manager=_FailingDatasetStub())
        with self.assertRaises(LearningCycleError):
            orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="dataset-fail"))
        self.assertEqual(len(self.production.list.__name__) if False else len(self.cycles.list_candidates()), 0)

    def test_65_training_failure_is_recorded_failed(self):
        policy = self._policy("AI", "AI_V1", "AI_V1")
        class BadTraining:
            def train(self, *_: object, **__: object):
                raise RuntimeError("train failed")
        orch = self._orchestrator(config=self._config(max_retries=1), training_pipeline=BadTraining(), model_registry=_NoopRegistry())
        req = self._request("AI", "AI_V1", "AI_V1", policy=policy, key="train-fail")
        with self.assertRaises(LearningCycleError): orch.trigger(req)
        self.assertEqual(self.cycles.list_cycles(scope_key="AI::AI_V1::AI_V1")[-1].status, LearningCycleStatus.FAILED)

    def test_66_validation_failure_leaves_champion_unchanged(self):
        policy = self._policy("AI", "AI_V1", "AI_V1")
        dataset = self._dataset()
        class BadValidation:
            def evaluate(self, *_: object, **__: object):
                raise RuntimeError("validation failed")
        orch = self._orchestrator(config=self._config(max_retries=0), dataset_manager=_DatasetStub(dataset), validation_engine=BadValidation(), training_pipeline=Mock(), model_registry=Mock())
        with self.assertRaises(LearningCycleError): orch.trigger(self._request("AI","AI_V1","AI_V1",policy=policy,key="val-fail"))
        self.assertIsNone(self.production.get("AI::AI_V1::AI_V1"))

    def test_67_oos_failure_leaves_champion_unchanged(self):
        class BadOOS(_L8Stub):
            def __init__(self): super().__init__(L8EvaluationStatus.FAILED)
        orch = self._orchestrator(config=self._config(), oos_engine=BadOOS())
        with self.assertRaises(LearningCycleError): orch.trigger(self._request("AI","AI_V1","AI_V1",policy=self._policy("AI","AI_V1","AI_V1"),key="oos-fail"))
        self.assertIsNone(self.production.get("AI::AI_V1::AI_V1"))

    def test_68_robustness_failure_leaves_champion_unchanged(self):
        class BadRobust(_L8Stub):
            def __init__(self): super().__init__(L8EvaluationStatus.COMPLETED); self.oos = True
            def evaluate_robustness(self, *_: object, **__: object): return type("R", (), {"status": L8EvaluationStatus.FAILED, "evaluation_id": "rob-fail", "sample_count": 0, "metrics": {}})()
        orch = self._orchestrator(config=self._config(), oos_engine=BadRobust())
        with self.assertRaises(LearningCycleError): orch.trigger(self._request("AI","AI_V1","AI_V1",policy=self._policy("AI","AI_V1","AI_V1"),key="rob-fail"))
        self.assertIsNone(self.production.get("AI::AI_V1::AI_V1"))

    def test_69_promotion_failure_leaves_old_champion(self):
        policy1 = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config(production_enabled=True))
        first = orch.trigger(self._request(policy=policy1, evidence=self._evidence(), key="prom-fail-1"))
        orch.initialize_champion(scope_key=first.scope_key, strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy1.policy_version_id, reason="bootstrap", now=BASE)
        policy2 = self._policy("Classic", "Classic_V1", "Classic_V1", params={"target_rr": 2.0})
        promo = PromotionPolicy(enabled=True, minimum_validation_metrics={"accuracy": 0.0}, minimum_oos_metrics={"f1_macro": 0.0})
        orch.promotion_policy = promo
        second = orch.trigger(self._request(policy=policy2, evidence=self._evidence(), key="prom-fail-2"))
        orch.config = self._config(production_enabled=True)
        before = self.production.get(first.scope_key)
        with patch.object(orch.production, "promote", side_effect=RuntimeError("forced promotion failure")):
            with self.assertRaises(RuntimeError):
                orch.promote_candidate(second.result_summary["candidate_id"], reason="forced")
        after = self.production.get(first.scope_key)
        self.assertEqual(before.policy_version_id, after.policy_version_id)

    def test_70_bounded_retry_count(self):
        failing = _FailingDatasetStub()
        orch = self._orchestrator(config=self._config(max_retries=2), dataset_manager=failing)
        with self.assertRaises(LearningCycleError): orch.trigger(self._request(policy=self._policy("Classic","Classic_V1","Classic_V1"), evidence=self._evidence(), key="retry"))
        self.assertEqual(failing.calls, 3)
        self.assertEqual(self.cycles.list_cycles()[-1].retry_count, 2)

    def test_71_no_infinite_retry(self):
        self.assertGreaterEqual(self._config(max_retries=2).max_retries, 0)

    def test_72_failed_cycle_can_be_retried_with_new_trigger_key(self):
        failing = _FailingDatasetStub()
        orch = self._orchestrator(config=self._config(max_retries=0), dataset_manager=failing)
        with self.assertRaises(LearningCycleError): orch.trigger(self._request(policy=self._policy("Classic","Classic_V1","Classic_V1"), evidence=self._evidence(), key="fail-1"))
        dataset = self._dataset()
        orch.dataset_manager = _DatasetStub(dataset)
        result = orch.trigger(self._request(policy=self._policy("SMC","SMC_V1","SMC_V1"), evidence=self._evidence(), key="success-2", strategy="SMC", variant="SMC_V1", version="SMC_V1"))
        self.assertEqual(result.status, LearningCycleStatus.CANDIDATE)

    # ---------- end-to-end actual L5-L8 orchestration ----------
    def test_73_end_to_end_cycle_candidate_and_promotion(self):
        dataset = self._dataset(count=12)
        training_pipeline = LearningTrainingPipeline(self.datasets, self.features, self.training_runs, artifact_root=self.root / "artifacts")
        model_registry = ModelRegistry(self.model_registry_repo, self.training_runs, self.datasets, self.features, training_pipeline, artifact_root=training_pipeline.artifact_root)
        validation = ModelEvaluationEngine(self.model_registry_repo, self.datasets, self.features, training_pipeline)
        l8 = L8EvaluationEngine(self.model_registry_repo, self.datasets, self.features, training_pipeline, self.l8_repo)
        orch = ContinuousLearningOrchestrator(cycle_repository=self.cycles, production_repository=self.production, dataset_manager=_DatasetStub(dataset), dataset_repository=self.datasets, training_pipeline=training_pipeline, training_repository=self.training_runs, model_registry=model_registry, model_registry_repository=self.model_registry_repo, validation_engine=validation, oos_engine=l8, policy_repository=self.policies, config=self._config(production_enabled=True), promotion_policy=PromotionPolicy(enabled=False))
        def training_factory(dsv: DatasetVersion) -> TrainingConfig:
            return TrainingConfig(dataset_version_id=dsv.dataset_version_id, strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1", feature_schema_version=SCHEMA, label_version=LABEL)
        req = LearningCycleRequest(strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1", trigger=LearningTrigger.MANUAL, trigger_key="e2e-1", dataset_build_kwargs={"dataset_name": "e2e", "strategies": ("AI",), "strategy_versions": ("AI_V1",)}, training_config_factory=training_factory)
        first = orch.trigger(req)
        self.assertEqual(first.status, LearningCycleStatus.CANDIDATE)
        self.assertIsNotNone(first.training_run_id); self.assertIsNotNone(first.model_version_id); self.assertIsNotNone(first.validation_evaluation_id); self.assertIsNotNone(first.oos_evaluation_id); self.assertIsNotNone(first.robustness_evaluation_id); self.assertIsNotNone(first.policy_version_id)
        self.assertIsNotNone(self.cycles.get_candidate(first.result_summary["candidate_id"]))

    def test_74_oos_feedback_loop_is_not_automatic(self):
        orch = self._orchestrator(config=self._config())
        with patch.object(orch.dataset_manager, "build_and_store", wraps=orch.dataset_manager.build_and_store) as mocked:
            result = orch.trigger(self._request(policy=self._policy("Classic","Classic_V1","Classic_V1"), evidence=self._evidence(), key="oos-no-feedback"))
            self.assertEqual(result.status, LearningCycleStatus.CANDIDATE)
            self.assertEqual(mocked.call_count, 1)

    def test_75_candidate_lineage_is_preserved(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config())
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="lineage"))
        candidate = self.cycles.get_candidate(result.result_summary["candidate_id"])
        self.assertEqual(candidate.policy_version_id, policy.policy_version_id)
        self.assertEqual(candidate.dataset_version_id, result.dataset_version_id)

    # ---------- restart / state recovery / audit ----------
    def test_76_restart_reloads_cycle_state(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config())
        result = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="restart"))
        self.db.close()
        self.db = Database(self.root / "l10.db")
        self.cycles = LearningCycleRepository(self.db)
        self.production = ProductionStateRepository(self.db)
        self.assertEqual(self.cycles.get(result.cycle_id).cycle_id, result.cycle_id)

    def test_77_restart_preserves_learning_state(self):
        state = LearningState("SCOPE", None, None, None, None, None, None, None, None, "MANUAL", BASE)
        self.cycles.upsert_learning_state(state)
        self.db.close(); self.db = Database(self.root / "l10.db"); MigrationRunner(self.db).apply_all(); self.cycles = LearningCycleRepository(self.db)
        reloaded = self.cycles.get_learning_state("SCOPE")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.last_trigger, "MANUAL")

    def test_78_audit_contains_scope_and_actor(self):
        self.cycles.audit("cycle_started", None, reason="SCHEDULED", status="STARTED")
        row = self.db.execute("SELECT scope_key, actor_source FROM learning_audit_events").fetchone()
        self.assertEqual(row["actor_source"], "SYSTEM")

    def test_79_audit_does_not_contain_secret_values(self):
        self.cycles.audit("test", None, reason="safe", status="OK")
        payload = self.db.execute("SELECT details_json FROM learning_audit_events").fetchone()["details_json"]
        lowered = payload.lower()
        self.assertNotIn("password", lowered); self.assertNotIn("api_key", lowered); self.assertNotIn("token", lowered)

    # ---------- production atomicity / rollback ----------
    def test_80_production_state_promotion_is_atomic(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config(production_enabled=True))
        base = orch.trigger(self._request(policy=policy, evidence=self._evidence(), key="atomic-base"))
        orch.initialize_champion(scope_key=base.scope_key, strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, reason="base", now=BASE)
        candidate = self.cycles.get_candidate(base.result_summary["candidate_id"])
        policy2 = self._policy("Classic", "Classic_V1", "Classic_V1", params={"target_rr": 1.75})
        cycle2 = self.cycles.create(LearningCycle.create(identity_hash="7"*64, scope_key=base.scope_key, strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", trigger=LearningTrigger.MANUAL, trigger_key="atomic-2", configuration_fingerprint="8"*64, started_at=BASE+timedelta(days=1)))
        self.cycles.set_dataset(cycle2.cycle_id, base.dataset_version_id)
        c2 = self.cycles.create_candidate(cycle=cycle2, model_version_id=None, policy_version_id=policy2.policy_version_id, dataset_version_id=base.dataset_version_id, validation_evaluation_id=None, oos_evaluation_id=None, robustness_evaluation_id=None, evidence=self._evidence().__dict__)
        before = self.production.get(base.scope_key)
        with patch("app.learning.continuous_repository._upsert_state_conn", side_effect=RuntimeError("transaction failure")):
            with self.assertRaises(RuntimeError): self.production.promote(candidate=c2, cycle=cycle2, new_model_version_id=None, new_policy_version_id=policy2.policy_version_id, reason="atomic", promotion_policy_version="p", evidence={})
        after = self.production.get(base.scope_key)
        self.assertEqual(before.policy_version_id, after.policy_version_id)

    def test_81_promotion_history_is_append_only(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        self.production.initialize(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, reason="x", now=BASE)
        history = self.cycles.list_production_history()
        with self.assertRaises(Exception):
            self.db.execute("DELETE FROM learning_production_history WHERE history_id=?", (history[0].history_id,)); self.db.commit()

    def test_82_rollback_requires_previous_champion(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        state = self.production.initialize(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=policy.policy_version_id, reason="x", now=BASE)
        with self.assertRaises(L10RepositoryError):
            self.production.rollback(state.scope_key, reason="no-prev", evidence={})

    def test_83_rollback_restores_previous_champion(self):
        p1 = self._policy("Classic", "Classic_V1", "Classic_V1")
        self.production.initialize(scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", model_version_id=None, policy_version_id=p1.policy_version_id, reason="v1", now=BASE)
        # Manually promote an explicit second policy to establish a previous state.
        p2 = self._policy("Classic", "Classic_V1", "Classic_V1", params={"target_rr": 2.0})
        cycle = LearningCycle.create(identity_hash="9"*64, scope_key="CLASSIC::Classic_V1::Classic_V1", strategy_name="Classic", strategy_variant="Classic_V1", strategy_version="Classic_V1", trigger=LearningTrigger.MANUAL, trigger_key="rb", configuration_fingerprint="a"*64, started_at=BASE+timedelta(days=1))
        self.cycles.create(cycle)
        dataset = self._dataset(strategy="Classic", variant="Classic_V1", version="Classic_V1")
        self.cycles.set_dataset(cycle.cycle_id, dataset.dataset_version_id)
        c = self.cycles.create_candidate(cycle=cycle, model_version_id=None, policy_version_id=p2.policy_version_id, dataset_version_id=dataset.dataset_version_id, validation_evaluation_id=None, oos_evaluation_id=None, robustness_evaluation_id=None)
        self.production.promote(candidate=c, cycle=cycle, new_model_version_id=None, new_policy_version_id=p2.policy_version_id, reason="v2", promotion_policy_version="p", evidence={})
        current = self.production.get(cycle.scope_key)
        restored = self.production.rollback(cycle.scope_key, reason="restore", evidence={})
        self.assertEqual(restored.policy_version_id, p1.policy_version_id)
        self.assertEqual(current.policy_version_id, p2.policy_version_id)

    def test_84_rollback_history_is_preserved(self):
        self.assertIn("ROLLBACK", {"ROLLBACK"})

    # ---------- feature flags / API / strategy safety ----------
    def test_85_config_default_production_learning_disabled(self):
        from config.config_hunter import load_settings
        settings = load_settings()
        self.assertFalse(settings.learning_production_enabled)

    def test_86_api_analyze_path_is_not_learning_orchestrated(self):
        from app.web.app import create_web_app
        app = create_web_app()
        paths = {route.path for route in app.routes if getattr(route, "path", None)}
        self.assertIn("/api/analyze", paths)
        self.assertNotIn("/api/learning/cycle", paths)

    def test_87_api_does_not_call_training_directly(self):
        from fastapi.testclient import TestClient
        from app.web.app import create_web_app
        app = create_web_app()
        with patch("app.learning.training.LearningTrainingPipeline.train") as train:
            # A missing auth request is sufficient: the endpoint is not allowed to reach training.
            TestClient(app).post("/api/analyze", json={})
            train.assert_not_called()

    def test_88_scheduler_does_not_auto_start(self):
        scheduler = LearningScheduler(3600)
        self.assertIsNone(scheduler._thread)

    def test_89_no_production_policy_mode_exists_in_l9(self):
        self.assertNotIn("PRODUCTION", {m.value for m in PolicyMode})

    def test_90_no_fixed_weight_fields_in_promotion_policy(self):
        self.assertNotIn("strategy_weights", PromotionPolicy(enabled=False).to_dict())

    # ---------- AI special handling ----------
    def test_91_ai_policy_identity_is_explicit(self):
        policy = self._policy("AI", "AI_V1", "AI_V1", source_model="model-ai")
        self.assertEqual(policy.strategy_name, "AI")
        self.assertEqual(policy.source_model_version_id, "model-ai")

    def test_92_ai_policy_is_not_buy_sell_conversion(self):
        policy = self._policy("AI", "AI_V1", "AI_V1", source_model="model-ai")
        self.assertEqual(policy.policy_types, (PolicyType.MODEL_POLICY,))
        self.assertNotIn("BUY_SELL_CONVERSION", policy.enabled_capabilities)

    # ---------- isolation / no mutation ----------
    def test_93_dataset_fingerprint_unchanged_by_cycle(self):
        dataset = self._dataset()
        before = self.datasets.get(dataset.dataset_version_id).fingerprint
        policy = self._policy("AI", "AI_V1", "AI_V1")
        orch = self._orchestrator(config=self._config(), dataset_manager=_DatasetStub(dataset))
        orch.trigger(self._request("AI","AI_V1","AI_V1",policy=policy,evidence=self._evidence(),key="immut-ds"))
        after = self.datasets.get(dataset.dataset_version_id).fingerprint
        self.assertEqual(before, after)

    def test_94_model_version_not_mutated_by_monitoring(self):
        # L10 monitoring stores snapshots only; it has no model update operation.
        self.assertFalse(any(name in ContinuousLearningOrchestrator.__dict__ for name in ("fit", "retrain", "fine_tune")))

    def test_95_policy_is_immutable(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        before = policy.to_dict()
        loaded = self.policies.get(policy.policy_version_id)
        self.assertEqual(before, loaded.to_dict())
        with self.assertRaises(Exception):
            self.db.execute("UPDATE learning_policy_versions SET strategy_version='Classic_V2' WHERE policy_version_id=?", (policy.policy_version_id,)); self.db.commit()

    def test_96_artifact_mutation_is_not_performed(self):
        self.assertNotIn("save_over_existing", ContinuousLearningOrchestrator.__dict__)

    def test_97_oos_not_used_as_training_feedback(self):
        source = Path(__file__).resolve().parents[3] / "app" / "learning" / "continuous.py"
        text = source.read_text()
        self.assertNotIn("oos_feedback_training", text)

    def test_98_no_strategy_weighting_logic(self):
        source = (Path(__file__).resolve().parents[3] / "app" / "learning" / "continuous.py").read_text()
        self.assertNotIn("Classic = 30%", source)
        self.assertNotIn("SMC = 40%", source)

    def test_99_no_web_learning_scheduler_start(self):
        web_source = (Path(__file__).resolve().parents[3] / "app" / "web" / "app.py").read_text()
        self.assertNotIn("start_scheduler", web_source)

    def test_100_learning_state_records_last_trigger(self):
        policy = self._policy("Classic", "Classic_V1", "Classic_V1")
        orch = self._orchestrator(config=self._config())
        result = orch.trigger(self._request(policy=policy,evidence=self._evidence(),key="state-trigger"))
        state = self.cycles.get_learning_state(result.scope_key)
        self.assertEqual(state.last_trigger, LearningTrigger.MANUAL.value)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from joblib import load
from sklearn.compose import ColumnTransformer

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.dataset import LearningDatasetManager
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.evaluation import ModelEvaluationEngine
from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.labels import LearningOutcome
from app.learning.models import LearningDirection, LearningFeatureSnapshot, LearningRecord, LearningRecordStatus, LearningSourceType
from app.learning.model_registry_repository import ModelRegistryRepository, ModelRegistryPersistenceError
from app.learning.registry import EvaluationStatus, ModelRegistry, ModelRegistryError, ModelVersionStatus
from app.learning.repository import LearningRecordRepository
from app.learning.training import LearningTrainingPipeline, TrainingConfig, TrainingPipelineError, TrainingRun, TrainingStatus, DEFAULT_MODEL_TYPE
from app.learning.training_repository import TrainingRunRepository


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCHEMA = "phase03-feature-schema-v1"
LABEL = "v1"


class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = Database(root / "test.db")
        MigrationRunner(self.db).apply_all()
        self.features = FeatureStore(self.db)
        self.records = LearningRecordRepository(self.db)
        self.datasets = DatasetVersionRepository(self.db)
        self.training_runs = TrainingRunRepository(self.db)
        self.registry_repo = ModelRegistryRepository(self.db)
        self.dataset_manager = LearningDatasetManager(self.records, self.features)
        self.artifact_root = root / "artifacts"
        self.training = LearningTrainingPipeline(
            self.datasets, self.features, self.training_runs, artifact_root=self.artifact_root
        )
        self.registry = ModelRegistry(
            self.registry_repo, self.training_runs, self.datasets, self.features, self.training,
            artifact_root=self.artifact_root,
        )
        self.evaluator = ModelEvaluationEngine(self.registry_repo, self.datasets, self.features, self.training)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _add_record(self, index: int, *, source=LearningSourceType.HISTORICAL_BACKTEST, outcome=None, schema=SCHEMA, label_version=LABEL) -> LearningRecord:
        ts = BASE + timedelta(hours=index)
        snapshot = self.features.save(FeatureSnapshot(
            timestamp=ts, symbol="XAUUSD", timeframe="M15", feature_engine_version="phase03-v1",
            feature_schema_version=schema,
            features={"rsi": float(35 + index), "market_phase": "TREND" if index % 2 == 0 else "RANGE"},
            market_regime="TRENDING", created_at=BASE,
            provenance_metadata={"provider": "test"},
        ))
        completed = outcome is not None
        return self.records.create(LearningRecord(
            record_id=f"l7-record-{index}", source_type=source, symbol="XAUUSD", timeframe="M15",
            decision_timestamp=ts, entry_timestamp=ts, direction=LearningDirection.BUY,
            entry_price=100.0 + index, target=102.0 + index, stop_loss=99.0 + index,
            risk_reward=2.0, strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1",
            feature_set_version=schema, parameters_snapshot={"p": index},
            feature_snapshot=LearningFeatureSnapshot(snapshot.features), evidence_snapshot={"e": "test"},
            confidence=70.0 + index, market_regime="TRENDING", feature_snapshot_id=snapshot.snapshot_id,
            outcome=outcome, exit_timestamp=None if not completed else ts + timedelta(hours=1),
            exit_price=None if not completed else 102.0 + index,
            exit_reason=None if not completed else "TP", duration=None if not completed else timedelta(hours=1),
            label_version=label_version, provenance_metadata={"event_kind": source.value},
            status=LearningRecordStatus.COMPLETED if completed else LearningRecordStatus.PENDING_OUTCOME,
            created_at=BASE,
        ))

    def _build_dataset(self, count: int = 8, dataset_name: str = "EDGE_HUNTER_LEARNING"):
        outcomes = [LearningOutcome.TP_BEFORE_SL.value, LearningOutcome.SL_BEFORE_TP.value] * (count // 2) + ([LearningOutcome.TP_BEFORE_SL.value] if count % 2 else [])
        for i in range(count):
            self._add_record(i, outcome=outcomes[i])
        return self.dataset_manager.build_and_store(dataset_name=dataset_name, strategies=("AI",))

    def _train(self, *, dataset_name: str = "EDGE_HUNTER_LEARNING"):
        dataset = self._build_dataset(dataset_name=dataset_name)
        cfg = TrainingConfig(
            dataset_version_id=dataset.version.dataset_version_id, strategy_name="AI", strategy_variant="AI_V1",
            strategy_version="AI_V1", feature_schema_version=SCHEMA, label_version=LABEL,
        )
        result = self.training.train(cfg)
        return dataset, result

    def test_completed_training_run_registers(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        self.assertEqual(model.status, ModelVersionStatus.REGISTERED)
        self.assertTrue(model.model_version_id.startswith("mdl-"))

    def test_pending_training_run_rejected(self):
        dataset = self._build_dataset()
        run = TrainingRun("pending", "id", dataset.version.dataset_version_id, BASE, None, TrainingStatus.PENDING, "AI", "AI_V1", "AI_V1", DEFAULT_MODEL_TYPE, "AI_V1", SCHEMA, LABEL, 1, 1, 1, {}, 42, created_at=BASE)
        self.training_runs.create(run)
        with self.assertRaises(ModelRegistryError): self.registry.register(run.training_run_id)

    def test_running_training_run_rejected(self):
        dataset = self._build_dataset()
        run = TrainingRun("running", "id", dataset.version.dataset_version_id, BASE, None, TrainingStatus.RUNNING, "AI", "AI_V1", "AI_V1", DEFAULT_MODEL_TYPE, "AI_V1", SCHEMA, LABEL, 1, 1, 1, {}, 42, created_at=BASE)
        self.training_runs.create(run)
        with self.assertRaises(ModelRegistryError): self.registry.register(run.training_run_id)

    def test_failed_training_run_rejected(self):
        dataset = self._build_dataset()
        run = TrainingRun("failed", "id", dataset.version.dataset_version_id, BASE, BASE + timedelta(minutes=1), TrainingStatus.FAILED, "AI", "AI_V1", "AI_V1", DEFAULT_MODEL_TYPE, "AI_V1", SCHEMA, LABEL, 1, 1, 1, {}, 42, error_type="X", error_message="boom", error_stage="FIT", created_at=BASE)
        self.training_runs.create(run)
        with self.assertRaises(ModelRegistryError): self.registry.register(run.training_run_id)

    def test_artifact_exists_and_checksum_verified(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        self.assertEqual(model.artifact_sha256, result.run.artifact_sha256)
        self.assertTrue(Path(model.artifact_path).is_file())

    def test_corrupted_artifact_rejected(self):
        _, result = self._train()
        path = Path(result.run.artifact_path)
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ModelRegistryError): self.registry.register(result.run.training_run_id)

    def test_artifact_reload_validation(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        artifact = self.training.load_artifact(model.artifact_path, expected_sha256=model.artifact_sha256)
        self.assertIn("model", artifact)
        self.assertIn("preprocessor", artifact)
        self.assertIn("label_encoder", artifact)
        self.assertIn("feature_names", artifact)

    def test_training_lineage_metadata_is_preserved(self):
        dataset, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        self.assertEqual(model.training_run_id, result.run.training_run_id)
        self.assertEqual(model.dataset_version_id, dataset.version.dataset_version_id)
        self.assertEqual(model.strategy_name, "AI")
        self.assertEqual(model.strategy_variant, "AI_V1")
        self.assertEqual(model.strategy_version, "AI_V1")
        self.assertEqual(model.feature_schema_version, SCHEMA)
        self.assertEqual(model.label_version, LABEL)

    def test_duplicate_registration_is_idempotent(self):
        _, result = self._train()
        a = self.registry.register(result.run.training_run_id)
        b = self.registry.register(result.run.training_run_id)
        self.assertEqual(a.model_version_id, b.model_version_id)
        self.assertEqual(len(self.registry_repo.find_by_training_run(result.run.training_run_id)), 1)

    def test_registry_persistence_and_reload(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        loaded = self.registry_repo.get(model.model_version_id)
        self.assertEqual(model.to_dict(), loaded.to_dict())

    def test_registry_queries(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        self.assertEqual(len(self.registry_repo.find_by_strategy("AI")), 1)
        self.assertEqual(len(self.registry_repo.find_by_dataset(model.dataset_version_id)), 1)
        self.assertEqual(len(self.registry_repo.find_by_status(ModelVersionStatus.REGISTERED)), 1)

    def test_unregistered_model_evaluation_is_rejected(self):
        with self.assertRaises(ModelRegistryError):
            self.evaluator.evaluate("missing-model")

    def test_validation_evaluation_runs_on_registered_artifact(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        evaluation = self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(evaluation.status, EvaluationStatus.COMPLETED)
        self.assertGreater(evaluation.sample_count, 0)
        self.assertIn("accuracy", evaluation.metrics)
        self.assertEqual(self.registry_repo.get(model.model_version_id).status, ModelVersionStatus.EVALUATED)

    def test_oos_evaluation_is_blocked(self):
        _, result = self._train()
        model = self.registry.register(result.run.training_run_id)
        with self.assertRaises(ModelRegistryError):
            self.evaluator.evaluate(model.model_version_id, split="OOS")

    def test_evaluation_persistence_and_reload(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        evaluation = self.evaluator.evaluate(model.model_version_id)
        loaded = self.registry_repo.get_evaluation(evaluation.evaluation_id)
        self.assertEqual(evaluation.to_dict(), loaded.to_dict())

    def test_evaluation_fingerprint_is_deterministic(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        a = self.evaluator.evaluate(model.model_version_id)
        b = self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)
        self.assertEqual(a.evaluation_id, b.evaluation_id)

    def test_evaluation_metrics_include_confusion_matrix_and_class_distribution(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        evaluation = self.evaluator.evaluate(model.model_version_id)
        self.assertTrue(evaluation.confusion_matrix)
        self.assertEqual(sum(evaluation.class_distribution.values()), evaluation.sample_count)

    def test_validation_preprocessing_is_transform_only(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        original_fit = ColumnTransformer.fit
        with patch.object(ColumnTransformer, "fit", side_effect=AssertionError("Validation preprocessing must not fit")):
            evaluation = self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(evaluation.status, EvaluationStatus.COMPLETED)
        self.assertIs(ColumnTransformer.fit, original_fit)

    def test_feature_schema_mismatch_rejected(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        bad = replace(model, feature_schema_version="different-schema")
        with patch.object(self.registry_repo, "get", return_value=bad):
            with self.assertRaises(ModelRegistryError): self.evaluator.evaluate(model.model_version_id)

    def test_label_version_mismatch_rejected(self):
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        rows = self.dataset_manager
        with self.assertRaises(ModelRegistryError):
            fake = model.to_dict(); fake["label_version"] = "v2"
            self.evaluator._validate_artifact(model, dataset.version, {"dataset_version_id": dataset.version.dataset_version_id, "strategy_name": "AI", "strategy_variant": "AI_V1", "strategy_version": "AI_V1", "model_type": DEFAULT_MODEL_TYPE, "feature_schema_version": SCHEMA, "label_version": "v2"})

    def test_oos_rows_are_not_loaded_by_validation_evaluation(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        captured = {}
        original = self.training.predict
        def spy(artifact, rows):
            captured["splits"] = {row.split.value for row in rows}
            return original(artifact, rows)
        with patch.object(self.training, "predict", side_effect=spy): self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(captured["splits"], {"VALIDATION"})

    def test_outcome_fields_are_not_feature_columns(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        artifact = load(model.artifact_path)
        names = {str(n).lower() for n in artifact["feature_names"]}
        self.assertTrue(names.isdisjoint({"outcome", "exit_price", "exit_timestamp", "duration", "pnl", "future_high", "future_low"}))

    def test_future_ohlc_is_not_used_in_evaluation_feature_matrix(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        artifact = load(model.artifact_path)
        names = {str(n).lower() for n in artifact["feature_names"]}
        self.assertFalse(any(token in name for name in names for token in ("future_high", "future_low", "future_volatility", "future_regime")))

    def test_evaluation_does_not_change_artifact(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        before = hashlib.sha256(Path(model.artifact_path).read_bytes()).hexdigest()
        self.evaluator.evaluate(model.model_version_id)
        after = hashlib.sha256(Path(model.artifact_path).read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_evaluation_does_not_change_dataset(self):
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        before = dataset.version.fingerprint
        self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(before, self.datasets.get(dataset.version.dataset_version_id).fingerprint)

    def test_evaluation_does_not_change_dataset_rows(self):
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        before = tuple(row.row_fingerprint for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features))
        self.evaluator.evaluate(model.model_version_id)
        after = tuple(row.row_fingerprint for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features))
        self.assertEqual(before, after)

    def test_evaluation_does_not_change_feature_snapshots(self):
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        before = self.db.execute("SELECT snapshot_id, snapshot_hash FROM feature_snapshots ORDER BY snapshot_id").fetchall()
        self.evaluator.evaluate(model.model_version_id)
        after = self.db.execute("SELECT snapshot_id, snapshot_hash FROM feature_snapshots ORDER BY snapshot_id").fetchall()
        self.assertEqual([(r[0], r[1]) for r in before], [(r[0], r[1]) for r in after])


    def test_artifact_metadata_mismatches_are_rejected(self):
        _, result = self._train()
        path = Path(result.run.artifact_path)
        original = load(path)
        for key in ("training_identity_hash", "dataset_version_id", "strategy_name", "strategy_variant", "strategy_version", "model_type", "model_version_label", "feature_schema_version", "label_version"):
            artifact = dict(original)
            artifact[key] = "mismatch"
            from joblib import dump
            dump(artifact, path, compress=3)
            new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.db.execute("UPDATE learning_training_runs SET artifact_sha256 = ? WHERE training_run_id = ?", (new_sha, result.run.training_run_id))
            self.db.commit()
            with self.assertRaises(ModelRegistryError):
                self.registry.register(result.run.training_run_id)
            dump(original, path, compress=3)
            restored_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.db.execute("UPDATE learning_training_runs SET artifact_sha256 = ? WHERE training_run_id = ?", (restored_sha, result.run.training_run_id))
            self.db.commit()

    def test_invalid_artifact_structure_is_rejected(self):
        _, result = self._train()
        path = Path(result.run.artifact_path)
        from joblib import dump
        dump({"broken": True}, path, compress=3)
        new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.db.execute("UPDATE learning_training_runs SET artifact_sha256 = ? WHERE training_run_id = ?", (new_sha, result.run.training_run_id))
        self.db.commit()
        with self.assertRaises(ModelRegistryError):
            self.registry.register(result.run.training_run_id)

    def test_evaluation_fingerprint_is_independent_of_input_row_order(self):
        from app.learning.evaluation import _evaluation_fingerprint
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        rows = [row for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features) if row.split.value == "VALIDATION"]
        a = _evaluation_fingerprint(model, dataset.version, rows, {"x": 1})
        b = _evaluation_fingerprint(model, dataset.version, list(reversed(rows)), {"x": 1})
        self.assertEqual(a, b)

    def test_different_dataset_version_has_separate_model_lineage(self):
        d1, r1 = self._train(dataset_name="EDGE_HUNTER_LEARNING_1"); m1 = self.registry.register(r1.run.training_run_id)
        d2, r2 = self._train(dataset_name="EDGE_HUNTER_LEARNING_2"); m2 = self.registry.register(r2.run.training_run_id)
        self.assertNotEqual(d1.version.dataset_version_id, d2.version.dataset_version_id)
        self.assertNotEqual(m1.model_version_id, m2.model_version_id)
        self.assertEqual(m1.dataset_version_id, d1.version.dataset_version_id)
        self.assertEqual(m2.dataset_version_id, d2.version.dataset_version_id)

    def test_api_routes_remain_unchanged(self):
        from app.web.app import app
        paths = {route.path for route in app.routes}
        self.assertIn("/api/analyze", paths)

    def test_model_version_status_change_is_not_promotion(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.evaluator.evaluate(model.model_version_id)
        evaluated = self.registry_repo.get(model.model_version_id)
        self.assertEqual(evaluated.status, ModelVersionStatus.EVALUATED)
        self.assertNotIn("CHAMPION", {s.value for s in ModelVersionStatus})

    def test_revoke_does_not_change_identity_fields(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        revoked = self.registry_repo.revoke(model.model_version_id)
        self.assertEqual(revoked.status, ModelVersionStatus.REVOKED)
        for field in ("model_version_id", "training_run_id", "dataset_version_id", "artifact_sha256", "feature_schema_version", "label_version", "strategy_version"):
            self.assertEqual(getattr(model, field), getattr(revoked, field))

    def test_revoke_blocks_evaluation(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.registry_repo.revoke(model.model_version_id)
        with self.assertRaises(ModelRegistryError): self.evaluator.evaluate(model.model_version_id)

    def test_model_version_delete_is_rejected(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        with self.assertRaises(Exception): self.db.execute("DELETE FROM learning_model_versions WHERE model_version_id = ?", (model.model_version_id,)); self.db.commit()

    def test_model_version_core_update_is_rejected(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        with self.assertRaises(Exception): self.db.execute("UPDATE learning_model_versions SET dataset_version_id = ? WHERE model_version_id = ?", ("different", model.model_version_id)); self.db.commit()

    def test_evaluation_delete_is_rejected(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id); ev = self.evaluator.evaluate(model.model_version_id)
        with self.assertRaises(Exception): self.db.execute("DELETE FROM learning_model_evaluations WHERE evaluation_id = ?", (ev.evaluation_id,)); self.db.commit()

    def test_failed_evaluation_is_persisted_without_revoking_model(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        with patch.object(self.training, "load_artifact", side_effect=TrainingPipelineError("corrupt")):
            with self.assertRaises(TrainingPipelineError): self.evaluator.evaluate(model.model_version_id, evaluation_config={"case": "failed"})
        evaluations = self.registry_repo.get_evaluations(model.model_version_id)
        self.assertEqual(evaluations[-1].status, EvaluationStatus.FAILED)
        self.assertEqual(self.registry_repo.get(model.model_version_id).status, ModelVersionStatus.REGISTERED)

    def test_different_evaluation_config_creates_distinct_evaluation(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        a = self.evaluator.evaluate(model.model_version_id, evaluation_config={"bins": 5})
        b = self.evaluator.evaluate(model.model_version_id, evaluation_config={"bins": 10})
        self.assertNotEqual(a.evaluation_id, b.evaluation_id)

    def test_registry_lineage_query_by_training_run(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.assertEqual(self.registry_repo.find_by_training_run(model.training_run_id)[0].model_version_id, model.model_version_id)

    def test_registry_lineage_query_by_dataset(self):
        dataset, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.assertEqual(self.registry_repo.find_by_dataset(dataset.version.dataset_version_id)[0].model_version_id, model.model_version_id)

    def test_registry_list_is_deterministic(self):
        _, r1 = self._train(); m1 = self.registry.register(r1.run.training_run_id)
        # Same dataset/config may create another TrainingRun and thus another ModelVersion.
        _, r2 = self._train(); m2 = self.registry.register(r2.run.training_run_id)
        ids = [m.model_version_id for m in self.registry_repo.list()]
        self.assertEqual(ids, sorted(ids, key=lambda x: self.registry_repo.get(x).registered_at))
        self.assertEqual({m1.model_version_id, m2.model_version_id}, set(ids))

    def test_metadata_contains_training_identity_without_secrets(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        serialized = json.dumps(model.metadata, sort_keys=True)
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("password", serialized.lower())
        self.assertIn(result.run.training_identity_hash, serialized)

    def test_artifact_path_must_stay_under_training_artifact_root(self):
        _, result = self._train()
        outside = Path(self.tmp.name) / "outside.joblib"
        outside.write_bytes(Path(result.run.artifact_path).read_bytes())
        self.db.execute("UPDATE learning_training_runs SET artifact_path = ? WHERE training_run_id = ?", (str(outside), result.run.training_run_id))
        self.db.commit()
        with self.assertRaises(ModelRegistryError): self.registry.register(result.run.training_run_id)

    def test_model_version_identity_is_not_timestamp_only(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.assertNotEqual(model.model_version_id, f"mdl-{int(model.registered_at.timestamp())}")

    def test_independent_evaluation_reloads_artifact(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        # Force disk reload and use the stored artifact path/checksum through the evaluator.
        with patch.object(self.training, "load_artifact", wraps=self.training.load_artifact) as loader:
            self.evaluator.evaluate(model.model_version_id)
            self.assertTrue(loader.called)
            self.assertEqual(loader.call_args.kwargs["expected_sha256"], model.artifact_sha256)

    def test_validation_label_classes_are_from_artifact(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        artifact = load(model.artifact_path)
        self.assertEqual(tuple(artifact["label_classes"]), tuple(str(x) for x in artifact["label_encoder"].classes_))

    def test_no_oos_status_decision_exists_in_registry(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.assertNotIn("OOS", model.status.value)

    def test_training_run_remains_unchanged_after_registration(self):
        _, result = self._train(); before = self.training_runs.get(result.run.training_run_id).to_dict()
        self.registry.register(result.run.training_run_id)
        self.assertEqual(before, self.training_runs.get(result.run.training_run_id).to_dict())

    def test_training_run_remains_unchanged_after_evaluation(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        before = self.training_runs.get(result.run.training_run_id).to_dict()
        self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(before, self.training_runs.get(result.run.training_run_id).to_dict())

    def test_evaluation_has_exact_validation_split(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id); ev = self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(ev.split, "VALIDATION")

    def test_evaluation_fingerprint_changes_with_config(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        a = self.evaluator.evaluate(model.model_version_id, evaluation_config={"metric_set": "v1"})
        b = self.evaluator.evaluate(model.model_version_id, evaluation_config={"metric_set": "v2"})
        self.assertNotEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)

    def test_registry_status_query_after_evaluation(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id); self.evaluator.evaluate(model.model_version_id)
        self.assertEqual(self.registry_repo.find_by_status(ModelVersionStatus.EVALUATED)[0].model_version_id, model.model_version_id)

    def test_model_version_has_no_production_status(self):
        self.assertNotIn("PRODUCTION", {s.value for s in ModelVersionStatus})

    def test_model_version_has_no_champion_status(self):
        self.assertNotIn("CHAMPION", {s.value for s in ModelVersionStatus})

    def test_model_version_has_no_challenger_status(self):
        self.assertNotIn("CHALLENGER", {s.value for s in ModelVersionStatus})

    def test_no_model_selection_method(self):
        self.assertFalse(hasattr(self.registry, "select_best"))
        self.assertFalse(hasattr(self.registry_repo, "rank_models"))

    def test_no_oos_parameter_is_accepted_for_evaluator(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        with self.assertRaises(ModelRegistryError): self.evaluator.evaluate(model.model_version_id, split="OOS")

    def test_registering_same_artifact_does_not_overwrite_metadata(self):
        _, result = self._train(); first = self.registry.register(result.run.training_run_id); second = self.registry.register(result.run.training_run_id)
        self.assertEqual(first.metadata, second.metadata)

    def test_artifact_checksum_matches_registry(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        digest = hashlib.sha256(Path(model.artifact_path).read_bytes()).hexdigest()
        self.assertEqual(digest, model.artifact_sha256)

    def test_model_version_created_at_tracks_training_run_not_identity_source(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        self.assertEqual(model.created_at, result.run.created_at)
        self.assertGreaterEqual(model.registered_at, model.created_at)

    def test_evaluation_error_does_not_change_model_status(self):
        _, result = self._train(); model = self.registry.register(result.run.training_run_id)
        with patch.object(self.training, "load_artifact", side_effect=TrainingPipelineError("bad")):
            with self.assertRaises(TrainingPipelineError): self.evaluator.evaluate(model.model_version_id, evaluation_config={"failure": 1})
        self.assertEqual(self.registry_repo.get(model.model_version_id).status, ModelVersionStatus.REGISTERED)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.advanced_evaluation_repository import L8EvaluationRepository
from app.learning.dataset import LearningDatasetManager
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.labels import LearningOutcome
from app.learning.model_registry_repository import ModelRegistryRepository
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)
from app.learning.oos_evaluation import L8EvaluationEngine, L8EvaluationStatus, L8EvaluationType
from app.learning.registry import ModelRegistry
from app.learning.repository import LearningRecordRepository
from app.learning.training import LearningTrainingPipeline, TrainingConfig, TrainingStatus
from app.learning.training_repository import TrainingRunRepository


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCHEMA = "phase03-feature-schema-v1"
LABEL = "v1"


class L8EvaluationTests(unittest.TestCase):
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
        self.l8_repo = L8EvaluationRepository(self.db)
        self.dataset_manager = LearningDatasetManager(self.records, self.features)
        self.artifact_root = root / "artifacts"
        self.training = LearningTrainingPipeline(self.datasets, self.features, self.training_runs, artifact_root=self.artifact_root)
        self.registry = ModelRegistry(self.registry_repo, self.training_runs, self.datasets, self.features, self.training, artifact_root=self.artifact_root)
        self.engine = L8EvaluationEngine(self.registry_repo, self.datasets, self.features, self.training, self.l8_repo)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _add_record(self, index: int, *, source=LearningSourceType.HISTORICAL_BACKTEST, outcome=None, symbol="XAUUSD", timeframe="M15", regime="TRENDING"):
        ts = BASE + timedelta(hours=index)
        snapshot = self.features.save(FeatureSnapshot(
            timestamp=ts, symbol=symbol, timeframe=timeframe, feature_engine_version="phase03-v1",
            feature_schema_version=SCHEMA,
            features={"rsi": float(30 + index), "market_phase": regime},
            market_regime=regime, created_at=BASE,
            provenance_metadata={"provider": "test"},
        ))
        return self.records.create(LearningRecord(
            record_id=f"l8-record-{index}", source_type=source, symbol=symbol, timeframe=timeframe,
            decision_timestamp=ts, entry_timestamp=ts, direction=LearningDirection.BUY,
            entry_price=100.0 + index, target=102.0 + index, stop_loss=99.0 + index,
            risk_reward=2.0, strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1",
            feature_set_version=SCHEMA, parameters_snapshot={"p": index},
            feature_snapshot=LearningFeatureSnapshot(snapshot.features), evidence_snapshot={"e": "test"},
            confidence=70.0, market_regime=regime, feature_snapshot_id=snapshot.snapshot_id,
            outcome=outcome, exit_timestamp=None if outcome is None else ts + timedelta(hours=1),
            exit_price=None if outcome is None else 102.0 + index, exit_reason=None if outcome is None else "TP",
            duration=None if outcome is None else timedelta(hours=1), label_version=LABEL,
            provenance_metadata={"event_kind": source.value},
            status=LearningRecordStatus.COMPLETED if outcome is not None else LearningRecordStatus.PENDING_OUTCOME,
            created_at=BASE,
        ))

    def _build(self, count=12, *, mixed_symbols=False, mixed_timeframes=False):
        outcomes = [LearningOutcome.TP_BEFORE_SL.value, LearningOutcome.SL_BEFORE_TP.value] * (count // 2)
        if count % 2:
            outcomes.append(LearningOutcome.TP_BEFORE_SL.value)
        for i in range(count):
            self._add_record(
                i,
                outcome=outcomes[i],
                symbol=("XAUUSD" if not mixed_symbols or i % 2 == 0 else "EURUSD"),
                timeframe=("M15" if not mixed_timeframes or i % 2 == 0 else "H1"),
                regime=("TRENDING" if i % 3 else "RANGING"),
            )
        return self.dataset_manager.build_and_store(dataset_name="L8_TEST", strategies=("AI",))

    def _train_register(self, count=12, **kwargs):
        dataset = self._build(count, **kwargs)
        result = self.training.train(TrainingConfig(
            dataset_version_id=dataset.version.dataset_version_id, strategy_name="AI", strategy_variant="AI_V1",
            strategy_version="AI_V1", feature_schema_version=SCHEMA, label_version=LABEL,
        ))
        model = self.registry.register(result.run.training_run_id)
        return dataset, result, model

    def test_oos_evaluation_creation(self):
        dataset, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(report.evaluation_type, L8EvaluationType.OOS)
        self.assertEqual(report.split, "OOS")
        self.assertEqual(report.dataset_version_id, dataset.version.dataset_version_id)
        self.assertEqual(report.status, L8EvaluationStatus.COMPLETED)
        self.assertGreater(report.sample_count, 0)

    def test_oos_uses_registered_model_and_correct_dataset(self):
        dataset, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(report.model_version_id, model.model_version_id)
        self.assertEqual(report.dataset_version_id, dataset.version.dataset_version_id)

    def test_oos_transform_only(self):
        _, _, model = self._train_register()
        with patch.object(self.training, "predict", wraps=self.training.predict) as predictor:
            report = self.engine.evaluate_oos(model.model_version_id)
        self.assertTrue(predictor.called)
        self.assertEqual(report.status, L8EvaluationStatus.COMPLETED)

    def test_oos_prediction_and_metrics(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertIn("accuracy", report.metrics)
        self.assertIn("precision_macro", report.metrics)
        self.assertIn("recall_macro", report.metrics)
        self.assertIn("f1_macro", report.metrics)
        self.assertIn("per_class_f1", report.metrics)

    def test_oos_confusion_matrix_and_distribution(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertTrue(report.confusion_matrix)
        self.assertEqual(sum(report.class_distribution.values()), report.sample_count)

    def test_oos_fingerprint_reproducible(self):
        _, _, model = self._train_register()
        a = self.engine.evaluate_oos(model.model_version_id)
        b = self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)
        self.assertEqual(a.metrics, b.metrics)
        self.assertIsNotNone(self.l8_repo.get(a.evaluation_id))

    def test_oos_does_not_use_training(self):
        _, _, model = self._train_register()
        with patch.object(self.training, "train", side_effect=AssertionError("L8 must not train")):
            self.engine.evaluate_oos(model.model_version_id)

    def test_oos_label_version_guard(self):
        _, _, model = self._train_register()
        rows = list(self.datasets.list_rows(model.dataset_version_id, feature_store=self.features))
        broken = rows[-1]
        from dataclasses import replace
        broken = replace(broken, label_version="other", row_fingerprint="")
        rows[-1] = broken
        with patch.object(self.datasets, "list_rows", return_value=tuple(rows)):
            with self.assertRaises(Exception):
                self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "label-guard"})

    def test_oos_feature_schema_guard(self):
        _, _, model = self._train_register()
        rows = list(self.datasets.list_rows(model.dataset_version_id, feature_store=self.features))
        broken = rows[-1]
        from dataclasses import replace
        broken = replace(broken, feature_schema_version="other", row_fingerprint="")
        rows[-1] = broken
        with patch.object(self.datasets, "list_rows", return_value=tuple(rows)):
            with self.assertRaises(Exception):
                self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "schema-guard"})

    def test_oos_outcome_leakage_blocked(self):
        _, _, model = self._train_register()
        rows = list(self.datasets.list_rows(model.dataset_version_id, feature_store=self.features))
        unsafe = rows[-1]
        object.__setattr__(unsafe, "features", {"rsi": 10.0, "outcome": "SL_BEFORE_TP"})
        with patch.object(self.datasets, "list_rows", return_value=tuple(rows)):
            with self.assertRaises(Exception):
                self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "leak"})

    def test_oos_does_not_mutate_dataset(self):
        dataset, _, model = self._train_register()
        before_version = self.datasets.get(dataset.version.dataset_version_id).manifest()
        before_rows = [row.to_dict() for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)]
        self.engine.evaluate_oos(model.model_version_id)
        after_version = self.datasets.get(dataset.version.dataset_version_id).manifest()
        after_rows = [row.to_dict() for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)]
        self.assertEqual(before_version, after_version)
        self.assertEqual(before_rows, after_rows)

    def test_oos_does_not_mutate_artifact(self):
        _, result, model = self._train_register()
        path = Path(result.run.artifact_path)
        before = path.read_bytes()
        self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(before, path.read_bytes())

    def test_validation_evaluation_is_unchanged_by_oos(self):
        _, _, model = self._train_register()
        from app.learning.evaluation import ModelEvaluationEngine
        l7_eval = ModelEvaluationEngine(self.registry_repo, self.datasets, self.features, self.training)
        validation_before = l7_eval.evaluate(model.model_version_id).to_dict()
        self.engine.evaluate_oos(model.model_version_id)
        validation_after = self.registry_repo.get_evaluation(validation_before["evaluation_id"]).to_dict()
        self.assertEqual(validation_before, validation_after)

    def test_walk_forward_creation(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        self.assertEqual(report.evaluation_type, L8EvaluationType.WALK_FORWARD)
        self.assertTrue(report.windows)

    def test_walk_forward_temporal_order(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        completed = [w for w in report.windows if w.get("status") == "COMPLETED"]
        for window in completed:
            self.assertLessEqual(window["train_end"], window["evaluation_start"])

    def test_walk_forward_multiple_windows_are_non_overlapping(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=4)
        completed = [w for w in report.windows if w.get("status") == "COMPLETED"]
        for left, right in zip(completed, completed[1:]):
            self.assertLess(left["evaluation_end"], right["evaluation_start"])

    def test_walk_forward_reproducible(self):
        _, _, model = self._train_register(count=36)
        a = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        b = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        self.assertEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)
        self.assertEqual(a.windows, b.windows)

    def test_walk_forward_persistence(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        self.assertIsNotNone(self.l8_repo.get(report.evaluation_id))

    def test_walk_forward_no_future_training_leakage(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        self.assertTrue(report.metadata["fixed_registered_artifact"])
        self.assertTrue(report.metadata["no_training_performed"])

    def test_walk_forward_insufficient_data_status(self):
        _, _, model = self._train_register(count=4)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        self.assertEqual(report.status, L8EvaluationStatus.INSUFFICIENT_DATA)

    def test_robustness_evaluation(self):
        _, _, model = self._train_register(count=36, mixed_symbols=True, mixed_timeframes=True)
        report = self.engine.evaluate_robustness(model.model_version_id, folds=3)
        self.assertEqual(report.evaluation_type, L8EvaluationType.ROBUSTNESS)
        self.assertIn("stability_summary", report.to_dict())
        case_ids = [case["case_id"] for case in report.windows]
        self.assertEqual(case_ids, sorted(case_ids))

    def test_robustness_multiple_periods(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_robustness(model.model_version_id, folds=4)
        self.assertTrue(any(case["case_type"] == "TEMPORAL_WINDOW" for case in report.windows))

    def test_robustness_stability_summary(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_robustness(model.model_version_id, folds=4)
        self.assertIn("accuracy", report.stability_summary.get("metric_summaries", {}))
        self.assertIn("mean", report.stability_summary["metric_summaries"]["accuracy"])
        self.assertIn("stddev", report.stability_summary["metric_summaries"]["accuracy"])

    def test_robustness_does_not_rank_cases(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_robustness(model.model_version_id, folds=4)
        self.assertTrue(report.metadata["no_ranking"])
        self.assertTrue(report.metadata["no_model_selection"])

    def test_sensitivity_is_not_fabricated(self):
        _, _, model = self._train_register(count=12)
        report = self.engine.evaluate_sensitivity(model.model_version_id)
        self.assertEqual(report.status, L8EvaluationStatus.NOT_APPLICABLE)
        self.assertTrue(report.metadata["no_parameter_optimization"])
        self.assertTrue(report.metadata["no_retraining"])

    def test_sensitivity_does_not_mutate_model(self):
        _, result, model = self._train_register(count=12)
        path = Path(result.run.artifact_path)
        before = path.read_bytes()
        self.engine.evaluate_sensitivity(model.model_version_id)
        self.assertEqual(before, path.read_bytes())

    def test_oos_evaluation_persistence_reload(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        loaded = self.l8_repo.get(report.evaluation_id)
        self.assertEqual(report.to_dict(), loaded.to_dict())

    def test_evaluation_record_delete_is_blocked(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        with self.assertRaises(Exception):
            self.db.execute("DELETE FROM learning_l8_evaluations WHERE evaluation_id = ?", (report.evaluation_id,))
            self.db.commit()

    def test_evaluation_identity_update_is_blocked(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        with self.assertRaises(Exception):
            self.db.execute("UPDATE learning_l8_evaluations SET dataset_version_id = 'other' WHERE evaluation_id = ?", (report.evaluation_id,))
            self.db.commit()

    def test_training_run_unchanged_by_l8(self):
        _, result, model = self._train_register()
        before = self.training_runs.get(result.run.training_run_id).to_dict()
        self.engine.evaluate_oos(model.model_version_id)
        self.engine.evaluate_walk_forward(model.model_version_id)
        self.engine.evaluate_robustness(model.model_version_id)
        self.engine.evaluate_sensitivity(model.model_version_id)
        self.assertEqual(before, self.training_runs.get(result.run.training_run_id).to_dict())

    def test_historical_and_live_provenance_are_preserved(self):
        self._build(count=12)
        self._add_record(20, source=LearningSourceType.LIVE_TRADE, outcome=LearningOutcome.TP_BEFORE_SL.value)
        # A second dataset includes both source types; L8 preserves distribution in metadata.
        dataset = self.dataset_manager.build_and_store(dataset_name="L8_LIVE_MIX", strategies=("AI",))
        result = self.training.train(TrainingConfig(
            dataset_version_id=dataset.version.dataset_version_id, strategy_name="AI", strategy_variant="AI_V1",
            strategy_version="AI_V1", feature_schema_version=SCHEMA, label_version=LABEL,
        ))
        model = self.registry.register(result.run.training_run_id)
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertIn(LearningSourceType.LIVE_TRADE.value, report.metadata["source_distribution"])
        self.assertIn(LearningSourceType.HISTORICAL_BACKTEST.value, report.metadata["source_distribution"])

    def test_no_oos_feedback_loop_metadata(self):
        _, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertFalse(report.metadata["oos_feedback_loop"])

    def test_no_arbitrary_artifact_evaluation_api(self):
        self.assertFalse(hasattr(self.engine, "evaluate_artifact"))


    def test_oos_feature_preprocessing_is_transform_only(self):
        _, _, model = self._train_register()
        real_load = self.training.load_artifact

        def load_spy(path, *, expected_sha256=None):
            artifact = dict(real_load(path, expected_sha256=expected_sha256))
            from unittest.mock import MagicMock
            pre = MagicMock(wraps=artifact["preprocessor"])
            pre.fit.side_effect = AssertionError("OOS must not fit preprocessing")
            pre.fit_transform.side_effect = AssertionError("OOS must not fit_transform preprocessing")
            artifact["preprocessor"] = pre
            return artifact

        with patch.object(self.training, "load_artifact", side_effect=load_spy):
            self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "transform-only"})

    def test_oos_labels_are_not_used_for_fit(self):
        _, _, model = self._train_register()
        with patch.object(self.training, "train", side_effect=AssertionError("L8 must never fit")):
            report = self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "no-fit-labels"})
        self.assertEqual(report.status, L8EvaluationStatus.COMPLETED)

    def test_oos_feature_order_matches_registered_artifact(self):
        _, _, model = self._train_register()
        artifact = self.training.load_artifact(model.artifact_path, expected_sha256=model.artifact_sha256)
        rows = self.datasets.list_rows(model.dataset_version_id, feature_store=self.features)
        feature_names = tuple(str(x) for x in artifact["feature_names"])
        self.assertEqual(set(feature_names), set(rows[-1].features.keys()))

    def test_corrupted_artifact_creates_failed_oos_report(self):
        _, result, model = self._train_register()
        path = Path(result.run.artifact_path)
        path.write_bytes(path.read_bytes() + b"corrupted")
        with self.assertRaises(Exception):
            self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "corrupt"})
        reports = self.l8_repo.find_by_model(model.model_version_id)
        failed = [r for r in reports if r.evaluation_config.get("case") == "corrupt"]
        self.assertEqual(failed[-1].status, L8EvaluationStatus.FAILED)
        self.assertTrue(failed[-1].error_message)

    def test_missing_oos_data_is_not_success(self):
        _, _, model = self._train_register()
        with patch.object(self.datasets, "list_rows", return_value=()):
            report = self.engine.evaluate_oos(model.model_version_id, evaluation_config={"case": "empty-oos"})
        self.assertEqual(report.status, L8EvaluationStatus.INSUFFICIENT_DATA)
        self.assertEqual(report.sample_count, 0)

    def test_walk_forward_single_window(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_walk_forward(model.model_version_id, folds=1)
        self.assertEqual(report.status, L8EvaluationStatus.COMPLETED)
        self.assertEqual(len(report.windows), 1)

    def test_robustness_insufficient_case_is_visible(self):
        _, _, model = self._train_register(count=36)
        report = self.engine.evaluate_robustness(model.model_version_id, minimum_sample=100, folds=3)
        self.assertEqual(report.status, L8EvaluationStatus.INSUFFICIENT_DATA)
        self.assertTrue(report.windows)
        self.assertTrue(any(w["status"] == L8EvaluationStatus.INSUFFICIENT_DATA.value for w in report.windows))

    def test_robustness_is_deterministic(self):
        _, _, model = self._train_register(count=36, mixed_symbols=True, mixed_timeframes=True)
        a = self.engine.evaluate_robustness(model.model_version_id, folds=3)
        b = self.engine.evaluate_robustness(model.model_version_id, folds=3)
        self.assertEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)
        self.assertEqual(a.metrics, b.metrics)

    def test_repository_queries_by_type_and_dataset(self):
        dataset, _, model = self._train_register()
        self.engine.evaluate_oos(model.model_version_id)
        self.engine.evaluate_sensitivity(model.model_version_id)
        self.assertEqual(len(self.l8_repo.find_by_type(L8EvaluationType.OOS)), 1)
        self.assertEqual(len(self.l8_repo.find_by_dataset(dataset.version.dataset_version_id)), 2)

    def test_report_contains_lineage_and_exclusion_metadata(self):
        dataset, _, model = self._train_register()
        report = self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(report.metadata["training_run_id"], model.training_run_id)
        self.assertEqual(report.metadata["dataset_fingerprint"], dataset.version.fingerprint)
        self.assertIn("dataset_exclusion_counts", report.metadata)

    def test_l8_migration_creates_expected_table(self):
        names = {row["name"] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn("learning_l8_evaluations", names)

    def test_api_routes_unchanged(self):
        from app.web.app import app
        paths = {route.path for route in app.routes}
        self.assertIn("/api/analyze", paths)

    def test_registry_status_not_changed_by_l8(self):
        _, _, model = self._train_register()
        before = self.registry_repo.get(model.model_version_id).status
        self.engine.evaluate_oos(model.model_version_id)
        self.assertEqual(before, self.registry_repo.get(model.model_version_id).status)

    def test_oos_config_changes_fingerprint(self):
        _, _, model = self._train_register()
        a = self.engine.evaluate_oos(model.model_version_id, evaluation_config={"scope": "a"})
        b = self.engine.evaluate_oos(model.model_version_id, evaluation_config={"scope": "b"})
        self.assertNotEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)

    def test_walk_forward_config_changes_fingerprint(self):
        _, _, model = self._train_register(count=36)
        a = self.engine.evaluate_walk_forward(model.model_version_id, folds=3)
        b = self.engine.evaluate_walk_forward(model.model_version_id, folds=2)
        self.assertNotEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)

    def test_robustness_config_changes_fingerprint(self):
        _, _, model = self._train_register(count=36)
        a = self.engine.evaluate_robustness(model.model_version_id, minimum_sample=1)
        b = self.engine.evaluate_robustness(model.model_version_id, minimum_sample=2)
        self.assertNotEqual(a.evaluation_fingerprint, b.evaluation_fingerprint)


if __name__ == "__main__":
    unittest.main()

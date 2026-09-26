"""L6 Training Pipeline tests: dataset isolation, leakage guards, artifacts and persistence."""

from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.dataset import DatasetBuildError, DatasetSplitPolicy, LearningDatasetManager
from app.learning.dataset_repository import DatasetVersionRepository
from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning.labels import LearningOutcome
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)
from app.learning.repository import LearningRecordRepository
from app.learning.training import (
    DEFAULT_MODEL_TYPE,
    LearningTrainingPipeline,
    PreparedDataset,
    TrainingConfig,
    TrainingPipelineError,
    TrainingStatus,
    _assert_features_safe,
    _assert_no_oos_in_training,
)
from app.learning.training_repository import TrainingRunRepository


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCHEMA = "features-v1"
LABEL = "v1"


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Database(self.root / "l6.db")
        MigrationRunner(self.db).apply_all()
        self.features = FeatureStore(self.db)
        self.records = LearningRecordRepository(self.db)
        self.datasets = DatasetVersionRepository(self.db)
        self.training_runs = TrainingRunRepository(self.db)
        self.manager = LearningDatasetManager(self.records, self.features)
        self.pipeline = LearningTrainingPipeline(
            self.datasets,
            self.features,
            self.training_runs,
            artifact_root=self.root / "artifacts",
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _add_record(
        self,
        index: int,
        *,
        outcome: str = LearningOutcome.TP_BEFORE_SL.value,
        source: LearningSourceType = LearningSourceType.HISTORICAL_BACKTEST,
        strategy: str = "AI",
        variant: str = "AI_V1",
        version: str = "AI_V1",
        schema: str = SCHEMA,
        label_version: str = LABEL,
    ) -> LearningRecord:
        decision = BASE + timedelta(hours=index)
        snapshot = FeatureSnapshot(
            timestamp=decision,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=schema,
            features={
                "rsi": float(35 + index * 8),
                "ema_distance": float((-2 if index % 2 == 0 else 2) + index * 0.1),
                "market_phase": "TREND" if index % 2 == 0 else "RANGE",
            },
            market_regime="TRENDING" if index % 2 == 0 else "RANGING",
            created_at=BASE,
            provenance_metadata={"provider": "test"},
        )
        snapshot = self.features.save(snapshot)
        record = LearningRecord(
            record_id=f"l6-record-{index}",
            source_type=source,
            symbol="XAUUSD",
            timeframe="M15",
            decision_timestamp=decision,
            entry_timestamp=decision,
            direction=LearningDirection.BUY,
            entry_price=100.0 + index,
            target=102.0 + index,
            stop_loss=99.0 + index,
            risk_reward=2.0,
            strategy_name=strategy,
            strategy_variant=variant,
            strategy_version=version,
            feature_set_version=schema,
            parameters_snapshot={"learning_feature": index},
            feature_snapshot=LearningFeatureSnapshot(snapshot.features),
            evidence_snapshot={"signal": "test"},
            confidence=70.0 + index,
            market_regime=snapshot.market_regime,
            feature_snapshot_id=snapshot.snapshot_id,
            outcome=outcome,
            exit_timestamp=decision + timedelta(hours=1),
            exit_price=102.0 + index,
            exit_reason="TP",
            duration=timedelta(hours=1),
            label_version=label_version,
            provenance_metadata={"event_kind": source.value},
            status=LearningRecordStatus.COMPLETED,
            created_at=BASE,
        )
        return self.records.create(record)

    def _build_dataset(self, *, count: int = 8, **filters):
        outcomes = [
            LearningOutcome.TP_BEFORE_SL.value,
            LearningOutcome.SL_BEFORE_TP.value,
            LearningOutcome.TP_BEFORE_SL.value,
            LearningOutcome.SL_BEFORE_TP.value,
            LearningOutcome.SL_BEFORE_TP.value,
            LearningOutcome.TP_BEFORE_SL.value,
            LearningOutcome.EXPIRED.value,
            LearningOutcome.TP_BEFORE_SL.value,
        ]
        for index in range(count):
            self._add_record(index, outcome=outcomes[index % len(outcomes)])
        result = self.manager.build_and_store(strategies=("AI",), **filters)
        return result

    def _config(self, dataset_id: str, **kwargs) -> TrainingConfig:
        values = {
            "dataset_version_id": dataset_id,
            "strategy_name": "AI",
            "strategy_variant": "AI_V1",
            "strategy_version": "AI_V1",
            "feature_schema_version": SCHEMA,
            "label_version": LABEL,
        }
        values.update(kwargs)
        return TrainingConfig(**values)

    def test_training_config_creation(self) -> None:
        config = self._config("dataset-1")
        self.assertEqual(config.model_type, DEFAULT_MODEL_TYPE)
        self.assertEqual(config.random_seed, 42)
        self.assertTrue(config.identity_hash())

    def test_non_ai_strategy_is_rejected(self) -> None:
        with self.assertRaises(TrainingPipelineError):
            TrainingConfig(
                dataset_version_id="d",
                strategy_name="SMC",
                strategy_variant="SMC_V1",
                strategy_version="SMC_V1",
                feature_schema_version=SCHEMA,
                label_version=LABEL,
            )

    def test_dataset_version_loading_and_split_counts(self) -> None:
        dataset = self._build_dataset()
        loaded = self.datasets.get(dataset.version.dataset_version_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.train_count, 4)
        self.assertEqual(loaded.validation_count, 2)
        self.assertEqual(loaded.oos_count, 2)

    def test_successful_training_uses_train_and_validation_only(self) -> None:
        dataset = self._build_dataset()
        seen: dict[str, int] = {}
        original_fit = self.pipeline._fit_model

        def spy_fit(x_train, y_train, config):
            seen["rows"] = len(y_train)
            return original_fit(x_train, y_train, config)

        self.pipeline._fit_model = spy_fit  # type: ignore[method-assign]
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        self.assertEqual(seen["rows"], dataset.version.train_count)
        self.assertEqual(result.run.status, TrainingStatus.COMPLETED)
        self.assertEqual(result.run.train_count, 4)
        self.assertEqual(result.run.validation_count, 2)
        self.assertEqual(result.run.oos_count, 2)
        self.assertIn("accuracy", result.train_metrics)
        self.assertIn("f1_macro", result.validation_metrics)

    def test_oos_never_enters_fit_data(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        self.assertEqual(result.run.oos_count, sum(row.split.value == "OOS" for row in rows))
        self.assertFalse(any(row.split.value == "OOS" for row in rows if row.record_id in {
            row.record_id for row in rows if row.split.value == "TRAIN"
        }))

    def test_oos_guard_fails_fast_on_contamination(self) -> None:
        dataset = self._build_dataset()
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        oos = next(row for row in rows if row.split.value == "OOS")
        with self.assertRaises(TrainingPipelineError):
            _assert_no_oos_in_training((oos,), (oos,))

    def test_feature_label_matrix_contains_only_snapshot_features(self) -> None:
        dataset = self._build_dataset()
        prepared = self.pipeline._prepare_dataset(dataset.version, self._config(dataset.version.dataset_version_id))
        self.assertEqual(prepared.feature_names, ("ema_distance", "market_phase", "rsi"))
        self.assertNotIn("outcome", prepared.feature_names)
        self.assertNotIn("exit_price", prepared.feature_names)
        self.assertEqual(len(prepared.y_train), dataset.version.train_count)

    def test_categorical_features_are_encoded_deterministically(self) -> None:
        dataset = self._build_dataset()
        prepared = self.pipeline._prepare_dataset(dataset.version, self._config(dataset.version.dataset_version_id))
        self.assertIn("market_phase", prepared.categorical_features)
        self.assertIn("rsi", prepared.numeric_features)
        first = self.pipeline._build_preprocessor(prepared).fit_transform(list(prepared.x_train_raw))
        second = self.pipeline._build_preprocessor(prepared).fit_transform(list(prepared.x_train_raw))
        self.assertEqual(first.shape, second.shape)
        self.assertEqual(first.tolist(), second.tolist())

    def test_preprocessing_feature_order_is_stored_in_artifact(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path, expected_sha256=result.run.artifact_sha256)
        self.assertEqual(tuple(artifact["feature_names"]), result.feature_names)
        self.assertEqual(artifact["feature_schema_version"], SCHEMA)
        self.assertEqual(artifact["label_version"], LABEL)
        self.assertIn("preprocessor", artifact)

    def test_label_version_and_feature_schema_are_validated(self) -> None:
        dataset = self._build_dataset()
        with self.assertRaises(TrainingPipelineError):
            self.pipeline.train(self._config(dataset.version.dataset_version_id, label_version="v2"))
        with self.assertRaises(TrainingPipelineError):
            self.pipeline.train(self._config(dataset.version.dataset_version_id, feature_schema_version="features-v2"))

    def test_strategy_scope_and_version_are_validated(self) -> None:
        dataset = self._build_dataset()
        with self.assertRaises(TrainingPipelineError):
            self.pipeline.train(self._config(dataset.version.dataset_version_id, strategy_version="AI_V2"))

    def test_training_run_is_persisted_and_reloadable(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        loaded = self.training_runs.get(result.run.training_run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, TrainingStatus.COMPLETED)
        self.assertEqual(loaded.artifact_sha256, result.run.artifact_sha256)
        self.assertTrue(Path(loaded.artifact_path).is_file())

    def test_artifact_reload_predictions_match(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path, expected_sha256=result.run.artifact_sha256)
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        validation = tuple(row for row in rows if row.split.value == "VALIDATION")
        predictions_1 = self.pipeline.predict(artifact, validation)
        predictions_2 = self.pipeline.predict(artifact, validation)
        self.assertEqual(predictions_1, predictions_2)
        self.assertEqual(len(predictions_1), len(validation))

    def test_reproducibility_same_dataset_config_seed(self) -> None:
        dataset = self._build_dataset()
        config = self._config(dataset.version.dataset_version_id)
        first = self.pipeline.train(config)
        second = self.pipeline.train(config)
        self.assertEqual(first.run.training_identity_hash, second.run.training_identity_hash)
        a1 = self.pipeline.load_artifact(first.artifact_path)
        a2 = self.pipeline.load_artifact(second.artifact_path)
        self.assertEqual(a1["feature_names"], a2["feature_names"])
        self.assertEqual(a1["label_classes"], a2["label_classes"])
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        validation = tuple(row for row in rows if row.split.value == "VALIDATION")
        self.assertEqual(self.pipeline.predict(a1, validation), self.pipeline.predict(a2, validation))
        self.assertEqual(a1["model"].coef_.tolist(), a2["model"].coef_.tolist())

    def test_different_config_changes_training_identity(self) -> None:
        dataset = self._build_dataset()
        a = self._config(dataset.version.dataset_version_id)
        b = self._config(dataset.version.dataset_version_id, training_parameters={"C": 0.25})
        self.assertNotEqual(a.identity_hash(), b.identity_hash())
        first = self.pipeline.train(a)
        second = self.pipeline.train(b)
        self.assertNotEqual(first.run.training_identity_hash, second.run.training_identity_hash)

    def test_different_dataset_version_changes_training_identity(self) -> None:
        first_dataset = self._build_dataset(count=8)
        config1 = self._config(first_dataset.version.dataset_version_id)
        # A new date-scope yields a different L5 DatasetVersion.
        later_dataset = self._build_dataset(count=8, start_timestamp=BASE + timedelta(hours=1))
        config2 = self._config(later_dataset.version.dataset_version_id)
        self.assertNotEqual(config1.identity_hash(), config2.identity_hash())

    def test_oos_rows_are_never_persisted_as_fit_metadata(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        self.assertEqual(result.run.metrics["oos_count"], dataset.version.oos_count)
        self.assertNotIn("oos_features", result.run.metrics)

    def test_validation_unavailable_is_recorded_without_fake_metrics(self) -> None:
        for index, outcome in enumerate([LearningOutcome.TP_BEFORE_SL.value, LearningOutcome.SL_BEFORE_TP.value, LearningOutcome.TP_BEFORE_SL.value, LearningOutcome.SL_BEFORE_TP.value]):
            self._add_record(index, outcome=outcome)
        policy = DatasetSplitPolicy(train_fraction=0.5, validation_fraction=0.0, version="no-validation-v1")
        dataset = self.manager.build_and_store(strategies=("AI",), split_policy=policy)
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        self.assertTrue(result.validation_unavailable)
        self.assertEqual(result.validation_metrics, {})

    def test_empty_train_fails_and_persists_failed_run(self) -> None:
        self._add_record(0, outcome=LearningOutcome.TP_BEFORE_SL.value)
        self._add_record(1, outcome=LearningOutcome.SL_BEFORE_TP.value)
        # The two-group L5 edge case yields a one-row TRAIN split, which is insufficient for classification.
        dataset = self.manager.build_and_store(strategies=("AI",))
        with self.assertRaises(TrainingPipelineError):
            self.pipeline.train(self._config(dataset.version.dataset_version_id))
        failed = self.training_runs.list(dataset_version_id=dataset.version.dataset_version_id)[-1]
        self.assertEqual(failed.status, TrainingStatus.FAILED)
        self.assertIsNotNone(failed.error_type)
        self.assertIsNotNone(failed.error_message)
        self.assertIsNotNone(failed.error_stage)

    def test_model_does_not_train_pending_or_invalid_records(self) -> None:
        pending_snapshot = FeatureSnapshot(
            timestamp=BASE + timedelta(hours=20),
            symbol="XAUUSD", timeframe="M15", feature_engine_version="phase03-v1",
            feature_schema_version=SCHEMA, features={"rsi": 40.0, "market_phase": "TREND"}, created_at=BASE,
        )
        pending_snapshot = self.features.save(pending_snapshot)
        pending = LearningRecord(
            record_id="pending-l6", source_type=LearningSourceType.HISTORICAL_BACKTEST,
            symbol="XAUUSD", timeframe="M15", decision_timestamp=pending_snapshot.timestamp,
            entry_timestamp=pending_snapshot.timestamp, direction=LearningDirection.BUY,
            entry_price=100.0, target=102.0, stop_loss=99.0, risk_reward=2.0,
            strategy_name="AI", strategy_variant="AI_V1", strategy_version="AI_V1",
            feature_set_version=SCHEMA, parameters_snapshot={}, feature_snapshot=LearningFeatureSnapshot(pending_snapshot.features),
            evidence_snapshot={"x": 1}, confidence=70.0, market_regime="TRENDING",
            feature_snapshot_id=pending_snapshot.snapshot_id, label_version=None,
            status=LearningRecordStatus.PENDING_OUTCOME, created_at=BASE,
        )
        self.records.create(pending)
        dataset = self._build_dataset(count=8)
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        self.assertTrue(all(row.record_id != "pending-l6" for row in rows))

    def test_dataset_version_and_rows_unchanged_after_training(self) -> None:
        dataset = self._build_dataset()
        before_version = self.datasets.get(dataset.version.dataset_version_id).manifest()
        before_rows = [row.to_dict() for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)]
        self.pipeline.train(self._config(dataset.version.dataset_version_id))
        after_version = self.datasets.get(dataset.version.dataset_version_id).manifest()
        after_rows = [row.to_dict() for row in self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)]
        self.assertEqual(before_version, after_version)
        self.assertEqual(before_rows, after_rows)

    def test_historical_and_live_provenance_are_retained(self) -> None:
        self._add_record(0, source=LearningSourceType.HISTORICAL_BACKTEST)
        self._add_record(1, source=LearningSourceType.HISTORICAL_BACKTEST, outcome=LearningOutcome.SL_BEFORE_TP.value)
        dataset = self.manager.build_and_store(strategies=("AI",), source_types=(LearningSourceType.HISTORICAL_BACKTEST,))
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        self.assertTrue(all(row.source_type == LearningSourceType.HISTORICAL_BACKTEST for row in rows))

        self._add_record(2, source=LearningSourceType.LIVE_TRADE, outcome=LearningOutcome.TP_BEFORE_SL.value)
        self._add_record(3, source=LearningSourceType.LIVE_TRADE, outcome=LearningOutcome.SL_BEFORE_TP.value)
        live_dataset = self.manager.build_and_store(strategies=("AI",), source_types=(LearningSourceType.LIVE_TRADE,))
        live_rows = self.datasets.list_rows(live_dataset.version.dataset_version_id, feature_store=self.features)
        self.assertTrue(all(row.source_type == LearningSourceType.LIVE_TRADE for row in live_rows))

    def test_mixed_feature_schema_is_rejected_by_dataset_version_before_training(self) -> None:
        self._add_record(0, schema="features-v1")
        self._add_record(1, schema="features-v2", outcome=LearningOutcome.SL_BEFORE_TP.value)
        with self.assertRaises(DatasetBuildError):
            self.manager.build_and_store(strategies=("AI",))

    def test_artifact_checksum_guard(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact_path = Path(result.artifact_path)
        artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
        with self.assertRaises(TrainingPipelineError):
            self.pipeline.load_artifact(artifact_path, expected_sha256=result.run.artifact_sha256)

    def test_train_validation_and_oos_rows_are_loaded_into_correct_groups(self) -> None:
        dataset = self._build_dataset()
        prepared = self.pipeline._prepare_dataset(dataset.version, self._config(dataset.version.dataset_version_id))
        self.assertEqual(len(prepared.train_rows), dataset.version.train_count)
        self.assertEqual(len(prepared.validation_rows), dataset.version.validation_count)
        self.assertEqual(len(prepared.oos_rows), dataset.version.oos_count)
        self.assertTrue(all(row.split.value == "TRAIN" for row in prepared.train_rows))
        self.assertTrue(all(row.split.value == "VALIDATION" for row in prepared.validation_rows))
        self.assertTrue(all(row.split.value == "OOS" for row in prepared.oos_rows))

    def test_validation_is_not_passed_to_model_fit(self) -> None:
        dataset = self._build_dataset()
        seen: dict[str, int] = {}
        original_fit = self.pipeline._fit_model

        def spy_fit(x_train, y_train, config):
            seen["fit_rows"] = len(y_train)
            return original_fit(x_train, y_train, config)

        self.pipeline._fit_model = spy_fit  # type: ignore[method-assign]
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        self.assertEqual(seen["fit_rows"], dataset.version.train_count)
        self.assertNotEqual(dataset.version.validation_count, seen["fit_rows"])
        self.assertEqual(result.run.validation_count, dataset.version.validation_count)

    def test_feature_matrix_has_expected_shape_for_train_and_validation(self) -> None:
        dataset = self._build_dataset()
        prepared = self.pipeline._prepare_dataset(dataset.version, self._config(dataset.version.dataset_version_id))
        self.assertEqual(len(prepared.x_train_raw), dataset.version.train_count)
        self.assertEqual(len(prepared.x_validation_raw), dataset.version.validation_count)
        self.assertTrue(all(len(row) == len(prepared.feature_names) for row in prepared.x_train_raw))
        self.assertTrue(all(len(row) == len(prepared.feature_names) for row in prepared.x_validation_raw))

    def test_training_parameters_are_captured_in_run_and_artifact(self) -> None:
        dataset = self._build_dataset()
        config = self._config(dataset.version.dataset_version_id, training_parameters={"C": 0.25, "max_iter": 750})
        result = self.pipeline.train(config)
        self.assertEqual(result.run.training_config["training_parameters"], {"C": 0.25, "max_iter": 750})
        artifact = self.pipeline.load_artifact(result.artifact_path)
        self.assertEqual(artifact["training_config"]["training_parameters"], {"C": 0.25, "max_iter": 750})

    def test_model_version_label_is_not_training_run_identity(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id, model_version_label="AI_V7"))
        self.assertEqual(result.run.model_version_label, "AI_V7")
        self.assertNotEqual(result.run.training_run_id, result.run.model_version_label)
        self.assertEqual(result.run.training_identity_hash, self._config(dataset.version.dataset_version_id, model_version_label="AI_V7").identity_hash())

    def test_random_seed_is_persisted_and_reloaded(self) -> None:
        dataset = self._build_dataset()
        config = self._config(dataset.version.dataset_version_id, random_seed=123)
        result = self.pipeline.train(config)
        loaded = self.training_runs.get(result.run.training_run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(result.run.random_seed, 123)
        self.assertEqual(loaded.random_seed, 123)
        artifact = self.pipeline.load_artifact(result.artifact_path)
        self.assertEqual(artifact["random_seed"], 123)

    def test_metrics_are_persisted_and_reloadable(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        loaded = self.training_runs.get(result.run.training_run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.metrics["train"]["accuracy"], result.train_metrics["accuracy"])
        self.assertEqual(loaded.metrics["validation"]["f1_macro"], result.validation_metrics["f1_macro"])

    def test_artifact_metadata_captures_schema_and_feature_order(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path)
        self.assertEqual(artifact["feature_schema_version"], SCHEMA)
        self.assertEqual(artifact["label_version"], LABEL)
        self.assertEqual(tuple(artifact["feature_names"]), tuple(sorted(artifact["feature_names"])))
        self.assertEqual(result.run.artifact_metadata["feature_count"], len(artifact["feature_names"]))

    def test_missing_dataset_version_fails_and_persists_failed_run(self) -> None:
        config = self._config("missing-dataset")
        with self.assertRaises(TrainingPipelineError) as ctx:
            self.pipeline.train(config)
        self.assertIn("DatasetVersion not found", str(ctx.exception))
        self.assertEqual(self.training_runs.list(dataset_version_id="missing-dataset"), [])

    def test_empty_train_split_fails_explicitly(self) -> None:
        dataset = self._build_dataset()
        original_prepare = self.pipeline._prepare_dataset

        def empty_train(dataset_arg, config):
            prepared = original_prepare(dataset_arg, config)
            return replace(prepared, train_rows=(), x_train_raw=(), y_train=())

        self.pipeline._prepare_dataset = empty_train  # type: ignore[method-assign]
        with self.assertRaises(TrainingPipelineError) as ctx:
            self.pipeline.train(self._config(dataset.version.dataset_version_id))
        self.assertIn("TRAIN split is empty", str(ctx.exception))
        failed = self.training_runs.list(dataset_version_id=dataset.version.dataset_version_id)[-1]
        self.assertEqual(failed.error_stage, "PREPARE_FEATURES")

    def test_dataset_version_is_immutable_at_database_level_after_training(self) -> None:
        dataset = self._build_dataset()
        self.pipeline.train(self._config(dataset.version.dataset_version_id))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "UPDATE learning_dataset_versions SET dataset_name = ? WHERE dataset_version_id = ?",
                ("tampered", dataset.version.dataset_version_id),
            )
        self.db.connection.rollback()
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "DELETE FROM learning_dataset_versions WHERE dataset_version_id = ?",
                (dataset.version.dataset_version_id,),
            )
        self.db.connection.rollback()

    def test_dataset_rows_are_immutable_at_database_level_after_training(self) -> None:
        dataset = self._build_dataset()
        self.pipeline.train(self._config(dataset.version.dataset_version_id))
        row = self.db.execute("SELECT record_id FROM learning_dataset_rows WHERE dataset_version_id = ? LIMIT 1", (dataset.version.dataset_version_id,)).fetchone()
        self.assertIsNotNone(row)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE learning_dataset_rows SET split = 'OOS' WHERE record_id = ? AND dataset_version_id = ?", (row["record_id"], dataset.version.dataset_version_id))
        self.db.connection.rollback()
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("DELETE FROM learning_dataset_rows WHERE record_id = ? AND dataset_version_id = ?", (row["record_id"], dataset.version.dataset_version_id))
        self.db.connection.rollback()

    def test_mixed_label_versions_are_rejected_before_training(self) -> None:
        self._add_record(0, label_version=LABEL)
        self._add_record(1, label_version="v2", outcome=LearningOutcome.SL_BEFORE_TP.value)
        with self.assertRaises(DatasetBuildError):
            self.manager.build_and_store(strategies=("AI",))

    def test_historical_and_live_provenance_are_stored_in_training_metrics(self) -> None:
        for index in range(4):
            self._add_record(index, source=LearningSourceType.HISTORICAL_BACKTEST, outcome=LearningOutcome.TP_BEFORE_SL.value if index % 2 == 0 else LearningOutcome.SL_BEFORE_TP.value)
        for index in range(4, 8):
            self._add_record(index, source=LearningSourceType.LIVE_TRADE, outcome=LearningOutcome.TP_BEFORE_SL.value if index % 2 == 0 else LearningOutcome.SL_BEFORE_TP.value)
        dataset = self.manager.build_and_store(strategies=("AI",))
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        distribution = result.run.metrics["provenance_distribution"]
        self.assertEqual(distribution[LearningSourceType.HISTORICAL_BACKTEST.value], 4)
        self.assertEqual(distribution[LearningSourceType.LIVE_TRADE.value], 4)

    def test_outcome_derived_feature_keys_are_rejected(self) -> None:
        class RowLike:
            features = {"rsi": 50.0, "future_high": 999.0}

        with self.assertRaises(TrainingPipelineError) as ctx:
            _assert_features_safe(RowLike())
        self.assertIn("forbidden feature field", str(ctx.exception))

    def test_future_ohlc_fields_never_appear_in_artifact_feature_names(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        forbidden_tokens = {"future_high", "future_low", "future_volatility", "future_regime", "exit_price", "outcome"}
        self.assertTrue(forbidden_tokens.isdisjoint(set(result.feature_names)))

    def test_training_run_preserves_strategy_metadata(self) -> None:
        for index in range(8):
            self._add_record(index, variant="AI_SIGNAL_V1", version="AI_SIGNAL_V1", outcome=LearningOutcome.TP_BEFORE_SL.value if index % 2 == 0 else LearningOutcome.SL_BEFORE_TP.value)
        dataset = self.manager.build_and_store(strategies=("AI",))
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id, strategy_variant="AI_SIGNAL_V1", strategy_version="AI_SIGNAL_V1"))
        self.assertEqual(result.run.strategy_name, "AI")
        self.assertEqual(result.run.strategy_variant, "AI_SIGNAL_V1")
        self.assertEqual(result.run.strategy_version, "AI_SIGNAL_V1")

    def test_training_run_repository_lists_runs_deterministically(self) -> None:
        dataset = self._build_dataset()
        first = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        second = self.pipeline.train(self._config(dataset.version.dataset_version_id, random_seed=7))
        runs = self.training_runs.list(dataset_version_id=dataset.version.dataset_version_id)
        self.assertEqual([run.training_run_id for run in runs], [first.run.training_run_id, second.run.training_run_id])
        self.assertEqual([run.status for run in runs], [TrainingStatus.COMPLETED, TrainingStatus.COMPLETED])

    def test_training_run_failure_stage_is_precise_for_fit_error(self) -> None:
        dataset = self._build_dataset()
        original_fit = self.pipeline._fit_model

        def explode(*args, **kwargs):
            raise RuntimeError("simulated fit failure")

        self.pipeline._fit_model = explode  # type: ignore[method-assign]
        try:
            with self.assertRaises(RuntimeError):
                self.pipeline.train(self._config(dataset.version.dataset_version_id))
        finally:
            self.pipeline._fit_model = original_fit  # type: ignore[method-assign]
        failed = self.training_runs.list(dataset_version_id=dataset.version.dataset_version_id)[-1]
        self.assertEqual(failed.status, TrainingStatus.FAILED)
        self.assertEqual(failed.error_stage, "FIT")
        self.assertEqual(failed.error_type, "RuntimeError")
        self.assertIn("simulated fit failure", failed.error_message or "")

    def test_training_run_completed_state_cannot_be_overwritten(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        current = self.training_runs.get(result.run.training_run_id)
        self.assertIsNotNone(current)
        with self.assertRaises(TrainingPipelineError):
            mutated = replace(current, metrics={"tampered": True})
            self.training_runs.mark_completed(mutated)

    def test_prediction_uses_persisted_feature_order_not_row_mapping_order(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path)
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        validation = tuple(row for row in rows if row.split.value == "VALIDATION")
        shuffled = tuple(
            replace(row, features={k: row.features[k] for k in reversed(list(row.features.keys()))})
            for row in validation
        )
        self.assertEqual(self.pipeline.predict(artifact, validation), self.pipeline.predict(artifact, shuffled))

    def test_sensitive_training_parameters_are_rejected(self) -> None:
        with self.assertRaises(TrainingPipelineError):
            self._config("dataset-1", training_parameters={"api_key": "secret"})

    def test_api_analyze_route_remains_present_without_training_integration(self) -> None:
        from app.web.app import app
        paths = {route.path for route in app.routes if getattr(route, "path", None)}
        self.assertIn("/api/analyze", paths)

    def test_training_does_not_change_dataset_fingerprint(self) -> None:
        dataset = self._build_dataset()
        before = self.datasets.get(dataset.version.dataset_version_id).fingerprint
        self.pipeline.train(self._config(dataset.version.dataset_version_id))
        after = self.datasets.get(dataset.version.dataset_version_id).fingerprint
        self.assertEqual(before, after)

    def test_training_artifact_contains_no_sensitive_dataset_metadata(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path)
        serialized = repr(artifact["training_config"])
        for secret_name in ("api_key", "password", "session", "subscription_code"):
            self.assertNotIn(secret_name, serialized.lower())

    def test_training_result_feature_names_match_artifact_feature_names(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path)
        self.assertEqual(result.feature_names, tuple(artifact["feature_names"]))

    def test_label_encoder_uses_train_labels_as_source_of_truth(self) -> None:
        dataset = self._build_dataset()
        result = self.pipeline.train(self._config(dataset.version.dataset_version_id))
        artifact = self.pipeline.load_artifact(result.artifact_path)
        rows = self.datasets.list_rows(dataset.version.dataset_version_id, feature_store=self.features)
        train_labels = sorted({row.outcome for row in rows if row.split.value == "TRAIN"})
        self.assertEqual(sorted(artifact["label_classes"]), train_labels)


if __name__ == "__main__":
    unittest.main()

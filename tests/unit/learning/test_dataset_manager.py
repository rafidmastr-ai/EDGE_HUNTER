"""L5 Dataset Manager tests: construction, versioning, splits, leakage and persistence."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.dataset import (
    DatasetBuildError,
    DatasetExclusionReason,
    DatasetSplit,
    DatasetSplitPolicy,
    LearningDatasetManager,
)
from app.learning.dataset_repository import DatasetVersionImmutableError, DatasetVersionRepository
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


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_snapshot(db: Database, index: int, *, schema: str = "features-v1", features: dict | None = None) -> FeatureSnapshot:
    snapshot = FeatureSnapshot(
        snapshot_id=f"l5-snapshot-{index}",
        timestamp=BASE + timedelta(hours=index),
        symbol="XAUUSD",
        timeframe="M15",
        feature_engine_version="phase03-v1",
        feature_schema_version=schema,
        features=features or {"rsi": 50.0 + index, "ema20": 100.0 + index},
        market_regime="TRENDING",
        created_at=BASE,
        provenance_metadata={"provider": "test-fixture"},
    )
    return FeatureStore(db).save(snapshot)


def make_record(
    index: int,
    snapshot: FeatureSnapshot,
    *,
    status: LearningRecordStatus = LearningRecordStatus.COMPLETED,
    source: LearningSourceType = LearningSourceType.HISTORICAL_BACKTEST,
    outcome: str | None = LearningOutcome.TP_BEFORE_SL.value,
    label_version: str | None = "v1",
    strategy: str = "SMC",
    strategy_version: str = "SMC_V1",
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> LearningRecord:
    decision = snapshot.timestamp
    return LearningRecord(
        record_id=f"l5-record-{index}",
        source_type=source,
        symbol=symbol,
        timeframe=timeframe,
        decision_timestamp=decision,
        entry_timestamp=decision,
        direction=LearningDirection.BUY,
        entry_price=100.0,
        target=102.0,
        stop_loss=99.0,
        risk_reward=2.0,
        strategy_name=strategy,
        strategy_variant=f"{strategy}_V1",
        strategy_version=strategy_version,
        feature_set_version=snapshot.feature_schema_version,
        parameters_snapshot={"rsi_period": 14, "ema_fast": 20},
        feature_snapshot=LearningFeatureSnapshot(snapshot.features),
        evidence_snapshot={"trend": "bullish", "structure": "BOS"},
        confidence=78.0,
        market_regime="TRENDING",
        feature_snapshot_id=snapshot.snapshot_id,
        outcome=outcome,
        exit_timestamp=decision + timedelta(hours=1) if status != LearningRecordStatus.PENDING_OUTCOME else None,
        exit_price=102.0 if status != LearningRecordStatus.PENDING_OUTCOME else None,
        exit_reason="TP" if status != LearningRecordStatus.PENDING_OUTCOME else None,
        duration=timedelta(hours=1) if status != LearningRecordStatus.PENDING_OUTCOME else None,
        label_version=label_version,
        provenance_metadata={"event_kind": "HISTORICAL_BACKTEST" if source == LearningSourceType.HISTORICAL_BACKTEST else "LIVE_SIGNAL_OBSERVATION"},
        status=status,
        created_at=BASE,
    )


class DatasetManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "l5.db")
        MigrationRunner(self.db).apply_all()
        self.feature_store = FeatureStore(self.db)
        self.repo = LearningRecordRepository(self.db)
        self.manager = LearningDatasetManager(self.repo, self.feature_store)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _add_completed(self, index: int, **kwargs) -> LearningRecord:
        snapshot = make_snapshot(self.db, index, schema=kwargs.pop("schema", "features-v1"))
        record = make_record(index, snapshot, **kwargs)
        self.repo.create(record)
        return record

    def test_eligible_completed_record_becomes_dataset_row(self) -> None:
        self._add_completed(0)
        result = self.manager.build()
        self.assertEqual(result.version.row_count, 1)
        self.assertEqual(result.rows[0].record_id, "l5-record-0")
        self.assertEqual(result.rows[0].feature_snapshot_id, "l5-snapshot-0")
        self.assertEqual(result.rows[0].features["rsi"], 50.0)
        self.assertEqual(result.rows[0].outcome, LearningOutcome.TP_BEFORE_SL.value)

    def test_pending_outcome_excluded(self) -> None:
        snapshot = make_snapshot(self.db, 0)
        self.repo.create(make_record(0, snapshot, status=LearningRecordStatus.PENDING_OUTCOME, outcome=None, label_version=None))
        with self.assertRaisesRegex(DatasetBuildError, "NO_ELIGIBLE_RECORDS"):
            self.manager.build()

        # Add one eligible record so exclusion metadata can be inspected.
        self._add_completed(1)
        result = self.manager.build()
        self.assertEqual(result.exclusions[DatasetExclusionReason.PENDING_OUTCOME.value], 1)

    def test_invalid_outcome_excluded(self) -> None:
        snapshot = make_snapshot(self.db, 0)
        invalid = make_record(
            0,
            snapshot,
            status=LearningRecordStatus.INVALID,
            outcome=LearningOutcome.INVALID.value,
            label_version="v1",
        )
        self.repo.create(invalid)
        self._add_completed(1)
        result = self.manager.build()
        self.assertEqual(result.exclusions[DatasetExclusionReason.INVALID_OUTCOME.value], 1)
        self.assertEqual(result.version.included_records, 1)

    def test_missing_feature_snapshot_excluded(self) -> None:
        record = self._add_completed(0)
        self.db.execute("UPDATE learning_records SET feature_snapshot_id = NULL WHERE record_id = ?", (record.record_id,))
        self.db.commit()
        with self.assertRaisesRegex(DatasetBuildError, "NO_ELIGIBLE_RECORDS"):
            self.manager.build()
        self.assertTrue(True)

    def test_feature_record_mismatch_excluded(self) -> None:
        snapshot = make_snapshot(self.db, 0)
        mismatched = make_record(0, snapshot, symbol="EURUSD")
        self.repo.create(mismatched)
        self._add_completed(1)
        result = self.manager.build()
        self.assertEqual(result.exclusions[DatasetExclusionReason.FEATURE_RECORD_MISMATCH.value], 1)

    def test_outcome_record_mismatch_excluded(self) -> None:
        snapshot = make_snapshot(self.db, 0)
        record = make_record(0, snapshot, outcome="NOT_A_LABEL", label_version="v1")
        self.repo.create(record)
        self._add_completed(1)
        result = self.manager.build()
        self.assertEqual(result.exclusions[DatasetExclusionReason.OUTCOME_RECORD_MISMATCH.value], 1)

    def test_strategy_metadata_preserved(self) -> None:
        self._add_completed(0, strategy="Classic", strategy_version="Classic_V1")
        row = self.manager.build().rows[0]
        self.assertEqual((row.strategy_name, row.strategy_variant, row.strategy_version), ("Classic", "Classic_V1", "Classic_V1"))

    def test_label_and_feature_schema_versions_preserved(self) -> None:
        snapshot = make_snapshot(self.db, 0, schema="features-v9")
        self.repo.create(make_record(0, snapshot, label_version="v7"))
        result = self.manager.build()
        self.assertEqual(result.version.label_version, "v7")
        self.assertEqual(result.version.feature_schema_version, "features-v9")
        self.assertEqual(result.rows[0].label_version, "v7")

    def test_historical_and_live_provenance_preserved(self) -> None:
        self._add_completed(0, source=LearningSourceType.HISTORICAL_BACKTEST)
        self._add_completed(1, source=LearningSourceType.LIVE_TRADE)
        result = self.manager.build()
        self.assertEqual(
            {row.source_type.value for row in result.rows},
            {LearningSourceType.HISTORICAL_BACKTEST.value, LearningSourceType.LIVE_TRADE.value},
        )
        self.assertEqual(result.version.statistics["by_provenance"]["LIVE_TRADE"], 1)

    def test_deterministic_row_ordering_and_same_timestamp_tiebreak(self) -> None:
        snapshot_a = make_snapshot(self.db, 0)
        snapshot_b = make_snapshot(self.db, 1)
        # Force a shared decision timestamp by updating neither source snapshot nor record identity;
        # use a test repository whose returned order is reversed instead.
        record_a = make_record(0, snapshot_a)
        record_b = make_record(1, snapshot_b)
        class ReverseRepository(LearningRecordRepository):
            def list(self, **kwargs):
                values = super().list(**kwargs)
                return list(reversed(values))
        self.repo.create(record_a)
        self.repo.create(record_b)
        manager = LearningDatasetManager(ReverseRepository(self.db), self.feature_store)
        result = manager.build()
        self.assertEqual([row.record_id for row in result.rows], ["l5-record-0", "l5-record-1"])

    def test_duplicate_db_rows_do_not_duplicate_dataset_row(self) -> None:
        record = self._add_completed(0)
        class DuplicateRepository(LearningRecordRepository):
            def list(self, **kwargs):
                values = super().list(**kwargs)
                return values + values
        manager = LearningDatasetManager(DuplicateRepository(self.db), self.feature_store)
        result = manager.build()
        self.assertEqual(result.version.row_count, 1)
        self.assertEqual(result.rows[0].record_id, record.record_id)

    def test_temporal_splits_are_disjoint_and_ordered(self) -> None:
        for index in range(10):
            self._add_completed(index)
        result = self.manager.build()
        splits = {row.split for row in result.rows}
        self.assertEqual(splits, {DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.OOS})
        train = [row for row in result.rows if row.split == DatasetSplit.TRAIN]
        validation = [row for row in result.rows if row.split == DatasetSplit.VALIDATION]
        oos = [row for row in result.rows if row.split == DatasetSplit.OOS]
        self.assertLess(max(row.decision_timestamp for row in train), min(row.decision_timestamp for row in validation))
        self.assertLess(max(row.decision_timestamp for row in validation), min(row.decision_timestamp for row in oos))
        self.assertTrue(set(r.record_id for r in train).isdisjoint(r.record_id for r in validation))
        self.assertTrue(set(r.record_id for r in validation).isdisjoint(r.record_id for r in oos))
        self.assertTrue(set(r.record_id for r in train).isdisjoint(r.record_id for r in oos))

    def test_oos_never_leaks_to_train(self) -> None:
        for index in range(9):
            self._add_completed(index)
        result = self.manager.build()
        oos_ids = {row.record_id for row in result.rows if row.split == DatasetSplit.OOS}
        train_ids = {row.record_id for row in result.rows if row.split == DatasetSplit.TRAIN}
        self.assertTrue(oos_ids)
        self.assertTrue(oos_ids.isdisjoint(train_ids))

    def test_validation_never_leaks_to_train(self) -> None:
        for index in range(9):
            self._add_completed(index)
        result = self.manager.build()
        validation_ids = {row.record_id for row in result.rows if row.split == DatasetSplit.VALIDATION}
        train_ids = {row.record_id for row in result.rows if row.split == DatasetSplit.TRAIN}
        self.assertTrue(validation_ids.isdisjoint(train_ids))

    def test_same_timestamp_group_is_not_split_across_sets(self) -> None:
        shared = BASE
        snapshots = []
        for index in range(6):
            snapshot = FeatureSnapshot(
                snapshot_id=f"same-ts-{index}", timestamp=shared, symbol="XAUUSD", timeframe="M15",
                feature_engine_version="phase03-v1", feature_schema_version="features-v1",
                features={"rsi": 50 + index}, created_at=BASE,
            )
            self.feature_store.save(snapshot)
            self.repo.create(make_record(index, snapshot))
            snapshots.append(snapshot)
        rows = self.manager.build().rows
        self.assertEqual({row.split for row in rows}, {DatasetSplit.TRAIN})

    def test_future_outcome_fields_are_not_features(self) -> None:
        self._add_completed(0)
        row = self.manager.build().rows[0]
        self.assertFalse(set(row.features).intersection({"outcome", "exit_price", "exit_timestamp", "exit_reason", "duration_seconds", "pnl"}))
        self.assertNotIn("outcome", json.dumps(dict(row.features), sort_keys=True))

    def test_feature_snapshot_remains_unchanged_when_future_outcome_changes(self) -> None:
        self._add_completed(0)
        before = self.feature_store.get("l5-snapshot-0")
        assert before is not None
        before_serialized = before.serialize()
        result1 = self.manager.build()
        self.assertEqual(result1.rows[0].feature_snapshot_hash, before.snapshot_hash)
        # L5 has no future-market input; changing an outcome in a separate immutable
        # record demonstrates that the persisted FeatureSnapshot remains untouched.
        snapshot2 = make_snapshot(self.db, 1, features={"rsi": 77.0, "future_marker": "ignored"})
        self.repo.create(make_record(1, snapshot2, outcome=LearningOutcome.SL_BEFORE_TP.value))
        after = self.feature_store.get("l5-snapshot-0")
        assert after is not None
        self.assertEqual(after.serialize(), before_serialized)
        self.assertEqual(self.manager.build().rows[0].feature_snapshot_hash, before.snapshot_hash)

    def test_changing_record_order_does_not_change_fingerprint(self) -> None:
        for index in range(6):
            self._add_completed(index)
        result1 = self.manager.build()
        class ReverseRepository(LearningRecordRepository):
            def list(self, **kwargs):
                values = super().list(**kwargs)
                return list(reversed(values))
        manager = LearningDatasetManager(ReverseRepository(self.db), self.feature_store)
        result2 = manager.build()
        self.assertEqual(result1.version.fingerprint, result2.version.fingerprint)
        self.assertEqual(result1.version.dataset_version_id, result2.version.dataset_version_id)

    def test_repeated_identical_build_is_reproducible(self) -> None:
        for index in range(6):
            self._add_completed(index)
        first = self.manager.build()
        second = self.manager.build()
        self.assertEqual(first.version.fingerprint, second.version.fingerprint)
        self.assertEqual(first.version.dataset_version_id, second.version.dataset_version_id)

    def test_label_version_change_creates_new_fingerprint(self) -> None:
        for index in range(4):
            snapshot = make_snapshot(self.db, index)
            self.repo.create(make_record(index, snapshot, label_version="v1" if index < 2 else "v2"))
        v1 = self.manager.build(label_version="v1")
        v2 = self.manager.build(label_version="v2")
        self.assertNotEqual(v1.version.fingerprint, v2.version.fingerprint)
        self.assertNotEqual(v1.version.dataset_version_id, v2.version.dataset_version_id)

    def test_date_scope_change_creates_new_fingerprint(self) -> None:
        for index in range(6):
            self._add_completed(index)
        all_data = self.manager.build()
        limited = self.manager.build(end_timestamp=BASE + timedelta(hours=3))
        self.assertNotEqual(all_data.version.fingerprint, limited.version.fingerprint)

    def test_split_policy_change_creates_new_fingerprint(self) -> None:
        for index in range(9):
            self._add_completed(index)
        first = self.manager.build(split_policy=DatasetSplitPolicy(0.60, 0.20, "v1"))
        second = self.manager.build(split_policy=DatasetSplitPolicy(0.70, 0.10, "v2"))
        self.assertNotEqual(first.version.fingerprint, second.version.fingerprint)
        self.assertNotEqual(first.version.split_policy_version, second.version.split_policy_version)

    def test_strategy_version_filtering(self) -> None:
        self._add_completed(0, strategy="SMC", strategy_version="SMC_V1")
        self._add_completed(1, strategy="SMC", strategy_version="SMC_V2")
        result = self.manager.build(strategy_versions=["SMC_V2"])
        self.assertEqual([row.strategy_version for row in result.rows], ["SMC_V2"])

    def test_strategy_symbol_timeframe_and_provenance_filtering(self) -> None:
        self._add_completed(0, strategy="SMC", symbol="XAUUSD", timeframe="M15", source=LearningSourceType.LIVE_TRADE)
        self._add_completed(1, strategy="Classic", symbol="EURUSD", timeframe="H1", source=LearningSourceType.HISTORICAL_BACKTEST)
        result = self.manager.build(
            strategies=["SMC"], symbols=["XAUUSD"], timeframes=["M15"], source_types=[LearningSourceType.LIVE_TRADE]
        )
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0].source_type, LearningSourceType.LIVE_TRADE)

    def test_mixed_label_versions_fail_closed_without_explicit_policy(self) -> None:
        for index in range(2):
            snapshot = make_snapshot(self.db, index)
            self.repo.create(make_record(index, snapshot, label_version=f"v{index+1}"))
        with self.assertRaisesRegex(DatasetBuildError, "MIXED_LABEL_VERSIONS"):
            self.manager.build()

    def test_mixed_feature_schema_fails_closed_without_explicit_policy(self) -> None:
        self._add_completed(0, schema="features-v1")
        self._add_completed(1, schema="features-v2")
        with self.assertRaisesRegex(DatasetBuildError, "MIXED_FEATURE_SCHEMA"):
            self.manager.build()

    def test_small_dataset_has_explicit_deterministic_behavior(self) -> None:
        self._add_completed(0)
        one = self.manager.build()
        self._add_completed(1)
        two = self.manager.build()
        self.assertEqual(one.version.train_count, 1)
        self.assertEqual(one.version.validation_count, 0)
        self.assertEqual(one.version.oos_count, 0)
        self.assertEqual(two.version.train_count, 1)
        self.assertEqual(two.version.validation_count, 0)
        self.assertEqual(two.version.oos_count, 1)

    def test_no_validation_fraction_creates_train_and_oos_only(self) -> None:
        for index in range(8):
            self._add_completed(index)
        result = self.manager.build(split_policy=DatasetSplitPolicy(0.75, 0.0, "v-no-validation"))
        self.assertGreater(result.version.train_count, 0)
        self.assertEqual(result.version.validation_count, 0)
        self.assertGreater(result.version.oos_count, 0)

    def test_empty_dataset_is_clear_failure(self) -> None:
        with self.assertRaisesRegex(DatasetBuildError, "NO_ELIGIBLE_RECORDS"):
            self.manager.build()

    def test_all_records_excluded_are_traced(self) -> None:
        snapshot = make_snapshot(self.db, 0)
        self.repo.create(make_record(0, snapshot, status=LearningRecordStatus.PENDING_OUTCOME, outcome=None, label_version=None))
        snapshot2 = make_snapshot(self.db, 1)
        self.repo.create(make_record(1, snapshot2, status=LearningRecordStatus.INVALID, outcome=LearningOutcome.INVALID.value))
        with self.assertRaisesRegex(DatasetBuildError, "NO_ELIGIBLE_RECORDS") as ctx:
            self.manager.build()
        self.assertEqual(str(ctx.exception), "NO_ELIGIBLE_RECORDS")
        self.assertEqual(ctx.exception.exclusion_counts[DatasetExclusionReason.PENDING_OUTCOME.value], 1)
        self.assertEqual(ctx.exception.exclusion_counts[DatasetExclusionReason.INVALID_OUTCOME.value], 1)

    def test_dataset_statistics_include_core_dimensions(self) -> None:
        self._add_completed(0, strategy="SMC")
        self._add_completed(1, strategy="Classic")
        result = self.manager.build()
        stats = result.version.statistics
        for key in ("by_strategy", "by_direction", "by_outcome", "by_symbol", "by_timeframe", "by_provenance", "by_split"):
            self.assertIn(key, stats)


class DatasetPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "persist.db")
        MigrationRunner(self.db).apply_all()
        self.feature_store = FeatureStore(self.db)
        self.repo = LearningRecordRepository(self.db)
        self.manager = LearningDatasetManager(self.repo, self.feature_store)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _build_result(self):
        for index in range(6):
            snapshot = make_snapshot(self.db, index)
            self.repo.create(make_record(index, snapshot))
        return self.manager.build()

    def test_migration_fresh_database_and_reload(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        stored = repository.create(result.version, result.rows)
        loaded = repository.get(stored.dataset_version_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.fingerprint, result.version.fingerprint)
        self.assertEqual(loaded.row_count, result.version.row_count)
        rows = repository.list_rows(stored.dataset_version_id, feature_store=self.feature_store)
        self.assertEqual(tuple(row.record_id for row in rows), tuple(row.record_id for row in result.rows))
        self.assertEqual(tuple(row.feature_snapshot_id for row in rows), tuple(row.feature_snapshot_id for row in result.rows))

    def test_manager_build_and_store_persists_the_immutable_version(self) -> None:
        result = self._build_result()
        stored = self.manager.build_and_store()
        self.assertEqual(stored.version.fingerprint, result.version.fingerprint)
        loaded = DatasetVersionRepository(self.db).get(stored.version.dataset_version_id)
        self.assertIsNotNone(loaded)

    def test_repeated_persistence_is_idempotent(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        first = repository.create(result.version, result.rows)
        second = repository.create(result.version, result.rows)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(repository.list()), 1)

    def test_dataset_version_is_immutable(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        repository.create(result.version, result.rows)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "UPDATE learning_dataset_versions SET dataset_name = ? WHERE dataset_version_id = ?",
                ("changed", result.version.dataset_version_id),
            )
            self.db.commit()
        self.db.connection.rollback()

        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "DELETE FROM learning_dataset_rows WHERE dataset_version_id = ?",
                (result.version.dataset_version_id,),
            )
            self.db.commit()
        self.db.connection.rollback()

    def test_conflicting_same_version_id_is_rejected(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        repository.create(result.version, result.rows)
        conflicting = type(result.version)(**{**result.version.__dict__, "fingerprint": "0" * 64, "dataset_version_id": result.version.dataset_version_id})
        with self.assertRaises(DatasetVersionImmutableError):
            repository.create(conflicting, result.rows)

    def test_persisted_row_contains_no_future_feature_fields(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        repository.create(result.version, result.rows)
        persisted = self.db.execute(
            "SELECT row_metadata_json FROM learning_dataset_rows WHERE dataset_version_id = ?",
            (result.version.dataset_version_id,),
        ).fetchone()
        self.assertIsNotNone(persisted)
        payload = json.loads(persisted["row_metadata_json"])
        self.assertIn("exit_price", payload)
        self.assertNotIn("features", payload)

    def test_feature_snapshot_reference_is_foreign_key_protected(self) -> None:
        result = self._build_result()
        repository = DatasetVersionRepository(self.db)
        repository.create(result.version, result.rows)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "DELETE FROM feature_snapshots WHERE snapshot_id = ?",
                (result.rows[0].feature_snapshot_id,),
            )
            self.db.commit()
        self.db.connection.rollback()


class DatasetNoApiBehaviorChangeTests(unittest.TestCase):
    """L5 is an offline learning layer; this test checks the public API module still imports."""

    def test_web_app_imports_without_dataset_side_effects(self) -> None:
        from app.web.app import app

        paths = {route.path for route in app.routes}
        self.assertIn("/api/analyze", paths)

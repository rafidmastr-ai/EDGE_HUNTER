from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from app.data.schema import CanonicalOHLC
from app.db.database import Database
from app.db.migrations import MIGRATIONS, MigrationRunner
from app.features.engine import FeatureEngine
from app.features.models import MarketAnalysisSnapshot
from app.learning.feature_store import (
    FEATURE_SCHEMA_VERSION,
    FeatureSnapshot,
    FeatureSnapshotError,
    FeatureSnapshotImmutableError,
    FeatureStore,
)
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningSourceType,
)
from app.learning.repository import LearningRecordRepository


UTC = timezone.utc


class ExampleState(Enum):
    ACTIVE = "ACTIVE"


def make_bars(count: int = 210, *, future_shift: Decimal = Decimal("0")) -> list[CanonicalOHLC]:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    bars: list[CanonicalOHLC] = []
    for index in range(count):
        shift = future_shift if index >= 121 else Decimal("0")
        close = Decimal("100") + Decimal(index) / Decimal("10") + shift
        open_ = close - Decimal("0.2")
        bars.append(
            CanonicalOHLC(
                timestamp=start + timedelta(minutes=index),
                open=open_,
                high=max(open_, close) + Decimal("0.4"),
                low=min(open_, close) - Decimal("0.3"),
                close=close,
                volume=Decimal(100 + index),
            )
        )
    return bars


def make_snapshot(*, snapshot_id: str = "snap-001", features: dict | None = None, created_at: datetime | None = None) -> FeatureSnapshot:
    timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    return FeatureSnapshot(
        snapshot_id=snapshot_id,
        timestamp=timestamp,
        symbol="XAUUSD",
        timeframe="M15",
        feature_engine_version="phase03-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features=features or {"price.close": 2500.25, "momentum.rsi": 57.5},
        market_regime="TRENDING",
        mtf_context={"H1": {"trend": "UP"}},
        created_at=created_at or timestamp,
        dataset_version="dataset-v1",
        provenance_metadata={"source": "feature-engine"},
    )


def make_learning_record(snapshot_id: str) -> LearningRecord:
    timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    return LearningRecord(
        record_id="lr-snapshot-link-001",
        source_type=LearningSourceType.HISTORICAL_BACKTEST,
        symbol="XAUUSD",
        timeframe="M15",
        decision_timestamp=timestamp,
        entry_timestamp=timestamp,
        direction=LearningDirection.BUY,
        entry_price=2500.0,
        target=2510.0,
        stop_loss=2495.0,
        risk_reward=2.0,
        strategy_name="SMC",
        strategy_variant="V1",
        strategy_version="smc-v1",
        feature_set_version=FEATURE_SCHEMA_VERSION,
        parameters_snapshot={"lookback": 5},
        feature_snapshot=LearningFeatureSnapshot({"price.close": 2500.25}),
        evidence_snapshot={"score": 0.8},
        confidence=75.0,
        market_regime="TRENDING",
        feature_snapshot_id=snapshot_id,
        created_at=timestamp,
    )


class FeatureSnapshotContractTests(unittest.TestCase):
    def test_creation_and_required_fields(self):
        snapshot = make_snapshot()
        self.assertEqual(snapshot.snapshot_id, "snap-001")
        self.assertEqual(snapshot.feature_schema_version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(snapshot.feature_engine_version, "phase03-v1")
        self.assertEqual(snapshot.market_regime, "TRENDING")
        self.assertEqual(snapshot.mtf_context["H1"]["trend"], "UP")

        with self.assertRaises(FeatureSnapshotError):
            FeatureSnapshot(
                timestamp=snapshot.timestamp,
                symbol="",
                timeframe="M15",
                feature_engine_version="phase03-v1",
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                features={},
            )

    def test_canonical_serialization_is_stable_and_round_trips(self):
        first = make_snapshot(created_at=datetime(2026, 9, 24, 11, 0, tzinfo=UTC))
        second = make_snapshot(created_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC))
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)

        restored = FeatureSnapshot.deserialize(first.serialize())
        self.assertEqual(restored.snapshot_hash, first.snapshot_hash)
        self.assertEqual(restored.to_dict(), first.to_dict())

    def test_hash_changes_for_meaningful_feature_change(self):
        original = make_snapshot()
        changed = make_snapshot(features={"price.close": 2500.26, "momentum.rsi": 57.5})
        self.assertNotEqual(original.snapshot_hash, changed.snapshot_hash)

    def test_key_order_does_not_change_hash(self):
        first = make_snapshot(features={"b": 2, "a": 1})
        second = make_snapshot(snapshot_id="snap-001", features={"a": 1, "b": 2})
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)
        self.assertEqual(first.serialize(), second.serialize())

    def test_special_values_are_handled_deterministically(self):
        timestamp = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
        original = FeatureSnapshot(
            timestamp=timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features={
                "decimal": Decimal("1.2300"),
                "captured_at": timestamp,
                "state": ExampleState.ACTIVE,
                "nested": {"values": (1, 2, 3), "optional": None},
            },
            created_at=timestamp,
        )
        restored = FeatureSnapshot.deserialize(original.serialize())
        self.assertEqual(restored.features["decimal"], Decimal("1.2300"))
        self.assertEqual(restored.features["captured_at"], timestamp)
        self.assertEqual(restored.features["state"], "ACTIVE")
        self.assertEqual(restored.features["nested"]["values"], (1, 2, 3))

    def test_outcome_and_sensitive_fields_are_rejected(self):
        for key in ("outcome", "exit_price", "pnl", "sl_hit", "api_key"):
            with self.subTest(key=key), self.assertRaises(FeatureSnapshotError):
                make_snapshot(features={key: "forbidden"})

    def test_immutable_feature_mapping(self):
        snapshot = make_snapshot()
        with self.assertRaises(TypeError):
            snapshot.features["price.close"] = 2600.0
        with self.assertRaises(TypeError):
            snapshot.mtf_context["H1"] = {"trend": "DOWN"}

    def test_from_market_analysis_snapshot_reuses_existing_features(self):
        source = MarketAnalysisSnapshot(
            timestamp=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
            symbol="xauusd",
            timeframe="M15",
            values={"price.close": 2500.0, "momentum.rsi": 55.0},
            metadata={
                "engine_version": "phase03-v1",
                "market_regime": "TRENDING",
                "mtf_context": {"H1": {"trend": "UP"}},
            },
            warmup_complete=True,
        )
        snapshot = FeatureSnapshot.from_market_analysis_snapshot(source)
        self.assertEqual(snapshot.features["price.close"], 2500.0)
        self.assertEqual(snapshot.feature_engine_version, "phase03-v1")
        self.assertEqual(snapshot.market_regime, "TRENDING")
        self.assertEqual(snapshot.mtf_context["H1"]["trend"], "UP")


class FeatureStorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "feature.db")
        MigrationRunner(self.db).apply_all()
        self.store = FeatureStore(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_migration_creates_feature_snapshot_schema(self):
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(feature_snapshots)").fetchall()}
        self.assertTrue(
            {
                "snapshot_id",
                "timestamp",
                "symbol",
                "timeframe",
                "feature_engine_version",
                "feature_schema_version",
                "features_json",
                "snapshot_hash",
                "created_at",
            }.issubset(columns)
        )
        learning_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(learning_records)").fetchall()}
        self.assertIn("feature_snapshot_id", learning_columns)

    def test_save_get_load_and_exists(self):
        snapshot = make_snapshot()
        saved = self.store.save(snapshot)
        self.assertEqual(saved.to_dict(), snapshot.to_dict())
        self.assertEqual(self.store.get(snapshot.snapshot_id).to_dict(), snapshot.to_dict())
        self.assertEqual(self.store.load(snapshot.snapshot_id).to_dict(), snapshot.to_dict())
        self.assertTrue(self.store.exists(snapshot_id=snapshot.snapshot_id))
        self.assertTrue(self.store.exists(snapshot_hash=snapshot.snapshot_hash))

    def test_duplicate_snapshot_save_is_idempotent(self):
        snapshot = make_snapshot()
        self.store.save(snapshot)
        self.assertEqual(self.store.save(snapshot).snapshot_id, snapshot.snapshot_id)
        equivalent = make_snapshot(snapshot_id="another-id")
        self.assertEqual(self.store.save(equivalent).snapshot_id, snapshot.snapshot_id)
        count = self.db.execute("SELECT COUNT(*) AS count FROM feature_snapshots").fetchone()["count"]
        self.assertEqual(count, 1)

    def test_existing_snapshot_id_cannot_be_overwritten(self):
        snapshot = make_snapshot()
        self.store.save(snapshot)
        changed = make_snapshot(snapshot_id=snapshot.snapshot_id, features={"price.close": 2600.0})
        with self.assertRaises(FeatureSnapshotImmutableError):
            self.store.save(changed)
        self.assertEqual(self.store.get(snapshot.snapshot_id).snapshot_hash, snapshot.snapshot_hash)

    def test_find_by_identity(self):
        snapshot = make_snapshot()
        self.store.save(snapshot)
        found = self.store.find_by_identity(
            symbol="xauusd",
            timeframe="M15",
            timestamp=snapshot.timestamp,
            feature_engine_version="phase03-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        self.assertEqual([item.snapshot_id for item in found], [snapshot.snapshot_id])

    def test_learning_record_can_reference_feature_snapshot(self):
        snapshot = self.store.save(make_snapshot())
        repo = LearningRecordRepository(self.db)
        record = make_learning_record(snapshot.snapshot_id)
        self.assertEqual(LearningRecord.from_dict(record.to_dict()).feature_snapshot_id, snapshot.snapshot_id)
        stored = repo.create(record)
        self.assertEqual(stored.feature_snapshot_id, snapshot.snapshot_id)
        row = self.db.execute(
            "SELECT feature_snapshot_id FROM learning_records WHERE record_id = ?",
            (stored.record_id,),
        ).fetchone()
        self.assertEqual(row["feature_snapshot_id"], snapshot.snapshot_id)

    def test_migration_is_idempotent_on_l1_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "existing-l1.db")
            try:
                runner = MigrationRunner(db)
                runner.apply_all()
                runner.apply_all()
                versions = [row["version"] for row in db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
                self.assertEqual(versions, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            finally:
                db.close()

    def test_l2_migration_upgrades_an_existing_l1_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "existing-l1.db")
            try:
                db.connect()
                db.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
                for migration in MIGRATIONS[:3]:
                    db.connection.executescript(migration.sql)
                    db.execute(
                        "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                        (migration.version, migration.name),
                    )
                    db.commit()
                before = [row["version"] for row in db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
                self.assertEqual(before, [1, 2, 3])

                runner = MigrationRunner(db)
                runner.apply_all()
                after = [row["version"] for row in db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
                self.assertEqual(after, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                self.assertIsNotNone(
                    db.execute(
                        "SELECT feature_snapshot_id FROM learning_records LIMIT 1"
                    ).description
                )
            finally:
                db.close()

    def test_snapshot_is_immutable_after_reload(self):
        snapshot = self.store.save(make_snapshot())
        reloaded = self.store.load(snapshot.snapshot_id)
        self.assertEqual(reloaded.snapshot_hash, snapshot.snapshot_hash)
        changed_features = {**snapshot.to_dict()["features"], "price.close": 2501.0}
        changed = make_snapshot(snapshot_id=snapshot.snapshot_id, features=changed_features)
        with self.assertRaises(FeatureSnapshotImmutableError):
            self.store.save(changed)


class LookAheadSafetyTests(unittest.TestCase):
    def test_snapshot_at_t_is_unchanged_when_future_bars_are_added_or_changed(self):
        bars = make_bars()
        future_changed_bars = make_bars(future_shift=Decimal("50"))
        target_index = 120

        engine = FeatureEngine()
        prefix_series = engine.compute(bars[: target_index + 1], "XAUUSD", "M15")
        extended_series = engine.compute(future_changed_bars, "XAUUSD", "M15")

        prefix_snapshot = FeatureSnapshot.from_market_analysis_snapshot(prefix_series.snapshots[-1])
        extended_snapshot = FeatureSnapshot.from_market_analysis_snapshot(extended_series.snapshots[target_index])

        self.assertEqual(prefix_snapshot.timestamp, extended_snapshot.timestamp)
        self.assertEqual(prefix_snapshot.features, extended_snapshot.features)
        self.assertEqual(prefix_snapshot.snapshot_hash, extended_snapshot.snapshot_hash)


if __name__ == "__main__":
    unittest.main()

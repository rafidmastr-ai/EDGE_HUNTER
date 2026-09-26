from datetime import datetime, timedelta, timezone
import tempfile
import unittest

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)
from app.learning.repository import (
    DuplicateLearningRecordError,
    LearningRecordRepository,
    LearningRecordStateError,
)


UTC = timezone.utc


def build_record(
    *,
    status: LearningRecordStatus = LearningRecordStatus.PENDING_OUTCOME,
    record_id: str = "lr-test-001",
    offset_minutes: int = 0,
) -> LearningRecord:
    completed = status == LearningRecordStatus.COMPLETED
    decision = datetime(2026, 9, 24, 10, 0, tzinfo=UTC) + timedelta(minutes=offset_minutes)
    return LearningRecord(
        record_id=record_id,
        source_type=LearningSourceType.HISTORICAL_BACKTEST,
        symbol="XAUUSD",
        timeframe="M15",
        decision_timestamp=decision,
        entry_timestamp=decision,
        direction=LearningDirection.BUY,
        entry_price=2500.0,
        target=2515.0,
        stop_loss=2490.0,
        risk_reward=1.5,
        strategy_name="SMC",
        strategy_variant="V1",
        strategy_version="smc-v1",
        feature_set_version="features-v1",
        parameters_snapshot={"lookback": 5},
        feature_snapshot=LearningFeatureSnapshot({"ema20": 2498.0, "rsi14": 57.5}),
        evidence_snapshot={"strategy_agreement": 0.8},
        confidence=72.5,
        market_regime="TRENDING",
        outcome="WIN" if completed else None,
        exit_timestamp=decision + timedelta(minutes=45) if completed else None,
        exit_price=2515.0 if completed else None,
        exit_reason="TARGET" if completed else None,
        duration=timedelta(minutes=45) if completed else None,
        dataset_version="dataset-v1",
        provenance_metadata={"backtest_run_id": "run-001"},
        status=status,
        created_at=decision,
    )


class LearningContractTests(unittest.TestCase):
    def test_learning_record_creation(self):
        record = build_record()
        self.assertEqual(record.source_type, LearningSourceType.HISTORICAL_BACKTEST)
        self.assertEqual(record.status, LearningRecordStatus.PENDING_OUTCOME)
        self.assertFalse(record.training_eligible)

    def test_validation_rejects_required_field_errors(self):
        with self.assertRaises(ValueError):
            build_record(record_id="")
        with self.assertRaises(ValueError):
            LearningRecord(
                **{**build_record().__dict__, "confidence": 101.0}
            )

    def test_source_type_and_lifecycle_validation(self):
        self.assertEqual(LearningSourceType("LIVE_TRADE"), LearningSourceType.LIVE_TRADE)
        with self.assertRaises(ValueError):
            LearningSourceType("BROKER_EXECUTION")
        with self.assertRaises(ValueError):
            LearningRecord(**{**build_record().__dict__, "status": LearningRecordStatus.PENDING_OUTCOME, "outcome": "WIN"})

    def test_live_trade_source_can_distinguish_signal_observation_from_execution_in_provenance(self):
        record = build_record()
        live = LearningRecord(**{
            **record.__dict__,
            "record_id": "lr-live-001",
            "source_type": LearningSourceType.LIVE_TRADE,
            "provenance_metadata": {"event_kind": "signal_observation"},
        })
        executed = LearningRecord(**{
            **record.__dict__,
            "record_id": "lr-live-002",
            "source_type": LearningSourceType.LIVE_TRADE,
            "provenance_metadata": {"event_kind": "broker_execution", "execution_reference": "exec-001"},
        })
        self.assertEqual(live.source_type, LearningSourceType.LIVE_TRADE)
        self.assertEqual(live.provenance_metadata["event_kind"], "signal_observation")
        self.assertEqual(executed.provenance_metadata["event_kind"], "broker_execution")

    def test_pending_outcome_is_not_training_eligible(self):
        self.assertFalse(build_record(status=LearningRecordStatus.PENDING_OUTCOME).training_eligible)

    def test_completed_can_be_training_eligible(self):
        self.assertTrue(build_record(status=LearningRecordStatus.COMPLETED).training_eligible)

    def test_invalid_and_excluded_are_not_training_eligible(self):
        self.assertFalse(build_record(status=LearningRecordStatus.INVALID).training_eligible)
        self.assertFalse(build_record(status=LearningRecordStatus.EXCLUDED).training_eligible)

    def test_feature_outcome_separation_and_sensitive_field_guard(self):
        with self.assertRaises(ValueError):
            LearningFeatureSnapshot({"outcome": "WIN"})
        with self.assertRaises(ValueError):
            LearningFeatureSnapshot({"api_key": "secret"})

    def test_serialization_round_trip(self):
        original = build_record()
        restored = LearningRecord.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertIn("feature_snapshot", original.to_dict())
        self.assertIn("outcome", original.to_dict())


class LearningPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(__import__("pathlib").Path(self.tmp.name) / "learning.db")
        MigrationRunner(self.db).apply_all()
        self.repo = LearningRecordRepository(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_migration_creates_learning_schema(self):
        names = {row["name"] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn("learning_records", names)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(learning_records)").fetchall()}
        self.assertTrue({"record_id", "identity_hash", "feature_snapshot_json", "outcome", "status"}.issubset(columns))

    def test_migration_is_idempotent(self):
        MigrationRunner(self.db).apply_all()
        rows = self.db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        self.assertEqual([row["version"] for row in rows], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def test_create_get_list_and_duplicate_protection(self):
        record = build_record()
        stored = self.repo.create(record)
        self.assertEqual(stored.to_dict(), record.to_dict())
        self.assertEqual(self.repo.get(record.record_id).to_dict(), record.to_dict())
        self.assertEqual(len(self.repo.list()), 1)
        self.assertTrue(self.repo.check_duplicate(record))
        self.assertEqual(self.repo.create(record).record_id, record.record_id)

        conflicting = build_record()
        with self.assertRaisesRegex(DuplicateLearningRecordError, "different learning record"):
            self.repo.create(LearningRecord(**{**conflicting.__dict__, "strategy_version": "other-version"}))

    def test_status_update_and_mark_completed(self):
        record = self.repo.create(build_record())
        with self.assertRaises(LearningRecordStateError):
            self.repo.update_status(record.record_id, LearningRecordStatus.COMPLETED)

        invalid = self.repo.update_status(record.record_id, LearningRecordStatus.INVALID)
        self.assertEqual(invalid.status, LearningRecordStatus.INVALID)
        with self.assertRaises(LearningRecordStateError):
            self.repo.mark_completed(
                record.record_id,
                outcome="WIN",
                exit_timestamp=record.decision_timestamp + timedelta(minutes=1),
                exit_price=2515.0,
                exit_reason="TARGET",
                duration=timedelta(minutes=1),
            )

        excluded = self.repo.create(build_record(record_id="lr-test-002", offset_minutes=1))
        self.assertEqual(
            self.repo.update_status(excluded.record_id, LearningRecordStatus.EXCLUDED).status,
            LearningRecordStatus.EXCLUDED,
        )
        with self.assertRaises(LearningRecordStateError):
            self.repo.mark_completed(
                excluded.record_id,
                outcome="WIN",
                exit_timestamp=excluded.decision_timestamp + timedelta(minutes=1),
                exit_price=2515.0,
                exit_reason="TARGET",
                duration=timedelta(minutes=1),
            )

        second = self.repo.create(build_record(record_id="lr-test-003", offset_minutes=2))
        completed = self.repo.mark_completed(
            second.record_id,
            outcome="WIN",
            exit_timestamp=second.decision_timestamp + timedelta(minutes=45),
            exit_price=2515.0,
            exit_reason="TARGET",
            duration=timedelta(minutes=45),
        )
        self.assertEqual(completed.status, LearningRecordStatus.COMPLETED)
        self.assertTrue(completed.training_eligible)
        self.assertEqual(completed.outcome, "WIN")

        completed_again = self.repo.mark_completed(
            second.record_id,
            outcome="WIN",
            exit_timestamp=second.decision_timestamp + timedelta(minutes=45),
            exit_price=2515.0,
            exit_reason="TARGET",
            duration=timedelta(minutes=45),
        )
        self.assertEqual(completed_again.to_dict(), completed.to_dict())

    def test_completed_record_persists_outcome_separately(self):
        record = self.repo.create(build_record())
        completed = self.repo.mark_completed(
            record.record_id,
            outcome="LOSS",
            exit_timestamp=record.decision_timestamp + timedelta(minutes=30),
            exit_price=2490.0,
            exit_reason="STOP",
            duration=timedelta(minutes=30),
        )
        row = self.db.execute(
            "SELECT feature_snapshot_json, outcome, exit_reason, duration_seconds, status FROM learning_records WHERE record_id = ?",
            (record.record_id,),
        ).fetchone()
        self.assertEqual(row["status"], "COMPLETED")
        self.assertEqual(row["outcome"], "LOSS")
        self.assertEqual(row["exit_reason"], "STOP")
        self.assertEqual(row["duration_seconds"], 1800)
        self.assertNotIn("outcome", row["feature_snapshot_json"])
        self.assertEqual(completed.feature_snapshot.to_dict()["ema20"], 2498.0)


if __name__ == "__main__":
    unittest.main()

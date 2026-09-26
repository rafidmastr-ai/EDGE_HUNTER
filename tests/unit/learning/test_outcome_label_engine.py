from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile

from app.backtest import BacktestConfig, IntrabarPolicy, TradeOutcome
from app.backtest.models import TradeRecord
from app.data.schema import CanonicalOHLC
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.feature_store import FeatureSnapshot, FeatureStore
from app.learning import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
    LearningOutcome,
    OutcomeLabelEngine,
    OutcomeResolutionState,
)
from app.learning.repository import LearningRecordRepository, LearningRecordStateError


UTC = timezone.utc
BASE = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)


def make_bar(index: int, *, close: float = 100.0, high: float | None = None, low: float | None = None) -> CanonicalOHLC:
    return CanonicalOHLC(
        timestamp=BASE + timedelta(minutes=index),
        open=Decimal(str(close)),
        high=Decimal(str(high if high is not None else close + 0.25)),
        low=Decimal(str(low if low is not None else close - 0.25)),
        close=Decimal(str(close)),
    )


def make_record(*, source: LearningSourceType = LearningSourceType.HISTORICAL_BACKTEST, direction: LearningDirection = LearningDirection.BUY, record_id: str = "lr-l4-001") -> LearningRecord:
    if direction == LearningDirection.BUY:
        entry, target, stop = 100.0, 102.0, 98.0
    else:
        entry, target, stop = 100.0, 98.0, 102.0
    return LearningRecord(
        record_id=record_id,
        source_type=source,
        symbol="XAUUSD",
        timeframe="M1",
        decision_timestamp=BASE,
        entry_timestamp=BASE,
        direction=direction,
        entry_price=entry,
        target=target,
        stop_loss=stop,
        risk_reward=2.0,
        strategy_name="SMC",
        strategy_variant="SMC_V1",
        strategy_version="SMC_V1",
        feature_set_version="features-v1",
        parameters_snapshot={"lookback": 20},
        feature_snapshot=LearningFeatureSnapshot({"rsi": 60.0, "ema20": 99.8}),
        evidence_snapshot={"structure": "BOS"},
        confidence=81.0,
        market_regime="TRENDING",
        feature_snapshot_id="snapshot-l4-001",
        dataset_version="dataset-v1",
        provenance_metadata={"event_kind": "HISTORICAL_BACKTEST_DECISION" if source == LearningSourceType.HISTORICAL_BACKTEST else "LIVE_SIGNAL_OBSERVATION"},
        created_at=BASE,
    )


class OutcomeLabelEngineTests(unittest.TestCase):
    def test_pending_outcome_is_accepted(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(make_record(), (make_bar(0), make_bar(1)))
        self.assertEqual(resolution.state, OutcomeResolutionState.NOT_ENOUGH_DATA_YET)
        self.assertEqual(resolution.source_type, LearningSourceType.HISTORICAL_BACKTEST)

    def test_buy_reaches_tp(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(
            make_record(), (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True
        )
        self.assertEqual(resolution.outcome, LearningOutcome.TP_BEFORE_SL)
        self.assertEqual(resolution.state, OutcomeResolutionState.COMPLETED)
        self.assertEqual(resolution.exit_reason, "TP")
        self.assertEqual(resolution.exit_timestamp, BASE + timedelta(minutes=1))
        self.assertEqual(resolution.exit_price, 102.0)
        self.assertEqual(resolution.bars_held, 1)
        self.assertEqual(resolution.duration, timedelta(minutes=1))

    def test_buy_reaches_sl(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(
            make_record(), (make_bar(0), make_bar(1, high=100.5, low=97.5)), data_complete=True
        )
        self.assertEqual(resolution.outcome, LearningOutcome.SL_BEFORE_TP)
        self.assertEqual(resolution.exit_reason, "SL")
        self.assertEqual(resolution.exit_price, 98.0)

    def test_sell_reaches_tp(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(
            make_record(direction=LearningDirection.SELL),
            (make_bar(0), make_bar(1, high=100.5, low=97.5)),
            data_complete=True,
        )
        self.assertEqual(resolution.outcome, LearningOutcome.TP_BEFORE_SL)
        self.assertEqual(resolution.exit_price, 98.0)

    def test_sell_reaches_sl(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(
            make_record(direction=LearningDirection.SELL),
            (make_bar(0), make_bar(1, high=102.5, low=99.5)),
            data_complete=True,
        )
        self.assertEqual(resolution.outcome, LearningOutcome.SL_BEFORE_TP)
        self.assertEqual(resolution.exit_price, 102.0)

    def test_expiry_uses_phase05_max_bars_semantics(self) -> None:
        engine = OutcomeLabelEngine(BacktestConfig(max_bars_in_trade=2))
        resolution = engine.evaluate(
            make_record(),
            (make_bar(0), make_bar(1), make_bar(2)),
            data_complete=False,
        )
        self.assertEqual(resolution.outcome, LearningOutcome.EXPIRED)
        self.assertEqual(resolution.exit_reason, "EXPIRY")
        self.assertEqual(resolution.bars_held, 2)
        self.assertEqual(resolution.duration, timedelta(minutes=2))

    def test_data_end_expiry_when_historical_input_is_complete(self) -> None:
        engine = OutcomeLabelEngine()
        resolution = engine.evaluate(
            make_record(), (make_bar(0), make_bar(1), make_bar(2)), data_complete=True
        )
        self.assertEqual(resolution.outcome, LearningOutcome.EXPIRED)
        self.assertEqual(resolution.exit_reason, "DATA_END")
        self.assertEqual(resolution.exit_timestamp, BASE + timedelta(minutes=2))

    def test_insufficient_future_data_does_not_become_expired(self) -> None:
        engine = OutcomeLabelEngine()
        resolution = engine.evaluate(
            make_record(), (make_bar(0), make_bar(1)), data_complete=False
        )
        self.assertEqual(resolution.state, OutcomeResolutionState.NOT_ENOUGH_DATA_YET)
        self.assertEqual(resolution.outcome, LearningOutcome.INVALID)
        self.assertIsNone(resolution.exit_timestamp)
        self.assertEqual(resolution.bars_held, 1)

    def test_entry_bar_is_excluded_from_evaluation(self) -> None:
        engine = OutcomeLabelEngine()
        bars = (
            make_bar(0, high=103.0, low=97.0),
            make_bar(1),
        )
        resolution = engine.evaluate(make_record(), bars, data_complete=False)
        self.assertEqual(resolution.state, OutcomeResolutionState.NOT_ENOUGH_DATA_YET)
        self.assertIsNone(resolution.exit_timestamp)

    def test_same_bar_ambiguity_follows_phase05_default_stop_first(self) -> None:
        resolution = OutcomeLabelEngine().evaluate(
            make_record(), (make_bar(0), make_bar(1, high=103.0, low=97.0)), data_complete=True
        )
        self.assertEqual(resolution.outcome, LearningOutcome.SL_BEFORE_TP)
        self.assertEqual(resolution.exit_reason, "TP_AND_SL_STOP_FIRST")

    def test_same_bar_target_first_is_supported(self) -> None:
        resolution = OutcomeLabelEngine(BacktestConfig(intrabar_policy=IntrabarPolicy.TARGET_FIRST)).evaluate(
            make_record(), (make_bar(0), make_bar(1, high=103.0, low=97.0)), data_complete=True
        )
        self.assertEqual(resolution.outcome, LearningOutcome.TP_BEFORE_SL)
        self.assertEqual(resolution.exit_reason, "TP_AND_SL_TARGET_FIRST")

    def test_same_bar_skip_maps_to_invalid_without_inventing_result(self) -> None:
        resolution = OutcomeLabelEngine(BacktestConfig(intrabar_policy=IntrabarPolicy.SKIP_TRADE)).evaluate(
            make_record(), (make_bar(0), make_bar(1, high=103.0, low=97.0)), data_complete=True
        )
        self.assertEqual(resolution.state, OutcomeResolutionState.INVALID)
        self.assertEqual(resolution.outcome, LearningOutcome.INVALID)
        self.assertEqual(resolution.backtest_outcome, TradeOutcome.AMBIGUOUS_SKIPPED.value)

    def test_invalid_trade_parameters_produce_invalid_resolution(self) -> None:
        invalid = LearningRecord(**{**make_record().__dict__, "target": 99.0})
        resolution = OutcomeLabelEngine().evaluate(invalid, (make_bar(0), make_bar(1)), data_complete=True)
        self.assertEqual(resolution.state, OutcomeResolutionState.INVALID)
        self.assertEqual(resolution.outcome, LearningOutcome.INVALID)
        self.assertIn("price geometry", resolution.exit_reason or "")

    def test_invalid_ohlc_produces_invalid_resolution(self) -> None:
        corrupted = CanonicalOHLC(
            timestamp=BASE + timedelta(minutes=1),
            open=Decimal("100"),
            high=Decimal("99"),
            low=Decimal("98"),
            close=Decimal("99"),
        )
        resolution = OutcomeLabelEngine().evaluate(make_record(), (make_bar(0), corrupted), data_complete=True)
        self.assertEqual(resolution.state, OutcomeResolutionState.INVALID)
        self.assertEqual(resolution.outcome, LearningOutcome.INVALID)
        self.assertIn("INVALID_OHLC", resolution.exit_reason or "")

    def test_spread_uses_phase05_exit_semantics(self) -> None:
        engine = OutcomeLabelEngine(BacktestConfig(spread=0.2))
        resolution = engine.evaluate(
            make_record(), (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True
        )
        self.assertAlmostEqual(resolution.exit_price or 0.0, 101.9)

    def test_historical_trade_outcome_adapter_reuses_existing_result(self) -> None:
        trade = TradeRecord(
            strategy_name="SMC",
            strategy_variant="SMC_V1",
            symbol="XAUUSD",
            timeframe="M1",
            direction="BUY",
            signal_timestamp=BASE,
            entry_timestamp=BASE,
            entry_price=100.0,
            stop_loss=98.0,
            target=102.0,
            planned_risk_reward=2.0,
            quantity=1.0,
            exit_timestamp=BASE + timedelta(minutes=1),
            exit_price=102.0,
            outcome=TradeOutcome.WIN,
            r_multiple=1.0,
            pnl_gross=2.0,
            commission=0.0,
            pnl_net=2.0,
            bars_held=1,
            exit_reason="TP",
        )
        resolution = OutcomeLabelEngine.from_trade_record(trade)
        self.assertEqual(resolution.outcome, LearningOutcome.TP_BEFORE_SL)
        self.assertEqual(resolution.backtest_outcome, "WIN")
        self.assertEqual(resolution.source_type, LearningSourceType.HISTORICAL_BACKTEST)

    def test_live_observation_uses_same_ohlc_path_without_broker_assumption(self) -> None:
        record = make_record(source=LearningSourceType.LIVE_TRADE, record_id="lr-live-l4")
        resolution = OutcomeLabelEngine().evaluate(
            record, (make_bar(0), make_bar(1, low=97.5, high=100.5)), data_complete=False
        )
        self.assertEqual(resolution.source_type, LearningSourceType.LIVE_TRADE)
        self.assertEqual(resolution.outcome, LearningOutcome.SL_BEFORE_TP)


class OutcomePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "l4.db")
        MigrationRunner(self.db).apply_all()
        self.feature_store = FeatureStore(self.db)
        self.feature_store.save(
            FeatureSnapshot(
                snapshot_id="snapshot-l4-001",
                timestamp=BASE,
                symbol="XAUUSD",
                timeframe="M1",
                feature_engine_version="phase03-v1",
                feature_schema_version="features-v1",
                features={"rsi": 60.0, "ema20": 99.8},
                market_regime="TRENDING",
                created_at=BASE,
            )
        )
        self.repo = LearningRecordRepository(self.db)
        self.engine = OutcomeLabelEngine()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_completed_record_contains_label_version_and_preserves_decision_snapshot(self) -> None:
        record = self.repo.create(make_record())
        before = record.to_dict()
        completed = self.engine.resolve_and_store(
            self.repo,
            record.record_id,
            (make_bar(0), make_bar(1, high=102.5, low=99.5)),
            data_complete=True,
        )
        self.assertEqual(completed.status, LearningRecordStatus.COMPLETED)
        self.assertEqual(completed.outcome, LearningOutcome.TP_BEFORE_SL.value)
        self.assertEqual(completed.label_version, "v1")
        self.assertEqual(completed.decision_timestamp, record.decision_timestamp)
        self.assertEqual(completed.strategy_version, record.strategy_version)
        self.assertEqual(completed.parameters_snapshot, record.parameters_snapshot)
        self.assertEqual(completed.confidence, record.confidence)
        self.assertEqual(completed.feature_snapshot.to_dict(), record.feature_snapshot.to_dict())
        self.assertEqual(before["decision_timestamp"], completed.to_dict()["decision_timestamp"])

    def test_reprocessing_completed_record_is_idempotent_and_cannot_reopen(self) -> None:
        record = self.repo.create(make_record())
        completed = self.engine.resolve_and_store(
            self.repo, record.record_id, (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True
        )
        again = self.engine.resolve_and_store(
            self.repo, record.record_id, (make_bar(0), make_bar(1, low=97.5, high=100.5)), data_complete=True
        )
        self.assertEqual(again.to_dict(), completed.to_dict())
        with self.assertRaises(LearningRecordStateError):
            self.repo.mark_completed(
                record.record_id,
                outcome=LearningOutcome.SL_BEFORE_TP.value,
                exit_timestamp=BASE + timedelta(minutes=1),
                exit_price=98.0,
                exit_reason="SL",
                duration=timedelta(minutes=1),
            )

    def test_insufficient_data_leaves_record_pending(self) -> None:
        record = self.repo.create(make_record(record_id="lr-l4-pending"))
        pending = self.engine.resolve_and_store(
            self.repo, record.record_id, (make_bar(0), make_bar(1)), data_complete=False
        )
        self.assertEqual(pending.status, LearningRecordStatus.PENDING_OUTCOME)
        self.assertIsNone(pending.outcome)
        self.assertIsNone(pending.label_version)

    def test_invalid_outcome_is_terminal_and_labeled(self) -> None:
        invalid = LearningRecord(**{**make_record(record_id="lr-l4-invalid").__dict__, "target": 99.0})
        stored = self.repo.create(invalid)
        resolved = self.engine.resolve_and_store(
            self.repo, stored.record_id, (make_bar(0), make_bar(1)), data_complete=True
        )
        self.assertEqual(resolved.status, LearningRecordStatus.INVALID)
        self.assertEqual(resolved.outcome, LearningOutcome.INVALID.value)
        self.assertEqual(resolved.label_version, "v1")

    def test_feature_snapshot_is_not_mutated_by_future_outcome(self) -> None:
        record = self.repo.create(make_record())
        original = record.feature_snapshot.to_dict()
        original_serialized = record.feature_snapshot.to_dict()
        self.engine.resolve_and_store(
            self.repo, record.record_id, (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True
        )
        stored = self.repo.get(record.record_id)
        self.assertEqual(stored.feature_snapshot.to_dict(), original)
        self.assertEqual(stored.feature_snapshot.to_dict(), original_serialized)
        self.assertNotIn("outcome", stored.feature_snapshot.to_dict())
        self.assertNotIn("exit_price", stored.feature_snapshot.to_dict())

    def test_future_bars_do_not_change_feature_snapshot_reference_or_hash(self) -> None:
        record = self.repo.create(make_record(record_id="lr-l4-leak"))
        feature_before = record.feature_snapshot.to_dict()
        resolution_one = self.engine.evaluate(record, (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True)
        resolution_two = self.engine.evaluate(
            record,
            (make_bar(0), make_bar(1, high=102.5, low=99.5), make_bar(2, high=120, low=80)),
            data_complete=True,
        )
        self.assertEqual(resolution_one.outcome, resolution_two.outcome)
        self.assertEqual(record.feature_snapshot.to_dict(), feature_before)
        self.assertEqual(record.feature_snapshot_id, "snapshot-l4-001")

    def test_label_version_survives_serialization_reload(self) -> None:
        record = self.repo.create(make_record(record_id="lr-l4-serialize"))
        completed = self.engine.resolve_and_store(
            self.repo, record.record_id, (make_bar(0), make_bar(1, high=102.5, low=99.5)), data_complete=True
        )
        restored = LearningRecord.from_dict(completed.to_dict())
        self.assertEqual(restored.label_version, "v1")
        self.assertEqual(restored.outcome, completed.outcome)
        self.assertEqual(restored.to_dict(), completed.to_dict())

    def test_migration_contains_label_version(self) -> None:
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(learning_records)").fetchall()}
        self.assertIn("label_version", columns)
        versions = [row["version"] for row in self.db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
        self.assertEqual(versions, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


if __name__ == "__main__":
    unittest.main()

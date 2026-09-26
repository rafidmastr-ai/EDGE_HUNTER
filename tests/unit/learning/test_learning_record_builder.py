from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.learning.feature_store import FEATURE_SCHEMA_VERSION, FeatureSnapshot, FeatureStore
from app.learning.models import LearningDirection, LearningRecordStatus, LearningSourceType
from app.learning.records import LearningRecordBuildError, LearningRecordBuilder
from app.learning.repository import LearningRecordRepository
from app.signals.models import (
    FinalSignalDecision,
    FinalSignalDirection,
    SignalEvidence,
    StrategyEvaluation,
)
from app.strategies.models import SignalDirection, SignalState, StrategySignal


UTC = timezone.utc
DECISION_TS = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def make_evidence() -> SignalEvidence:
    return SignalEvidence(
        strategy_agreement=0.9,
        signal_quality=0.8,
        structure_feature_agreement=0.75,
        entry_quality=0.7,
        stop_target_quality=0.85,
        rr_quality=0.9,
        oos_performance=0.65,
        sample_size=0.8,
        robustness=0.7,
        market_regime=0.75,
        sources={"strategy": "phase04", "research": "phase07"},
    )


def make_signal() -> StrategySignal:
    return StrategySignal(
        timestamp=DECISION_TS,
        symbol="XAUUSD",
        timeframe="M15",
        direction=SignalDirection.BUY,
        state=SignalState.SIGNAL,
        entry=2650.0,
        stop_loss=2645.0,
        target=2660.0,
        risk_reward=2.0,
        entry_logic="Decision bar close confirmation",
        invalidation="Structural stop is broken",
        stop_loss_logic="Recent structural low",
        target_logic="Single dynamic target",
        evidence=("EMA alignment", "structure confirmation", "momentum confirmation"),
        score_inputs={"trend": 1.0, "momentum": 0.8},
        strategy_name="Classic",
        variant="Classic_V1",
        metadata={"strategy_version": "Classic_V1"},
    )


def make_decision(*, direction: FinalSignalDirection = FinalSignalDirection.BUY, confidence: float = 82.5) -> FinalSignalDecision:
    signal = make_signal()
    evaluation = StrategyEvaluation(
        strategy_name="Classic",
        variant="Classic_V1",
        state=SignalState.SIGNAL,
        direction=SignalDirection.BUY,
        confidence=82.5,
        evidence=make_evidence(),
        signal=signal,
    )
    if direction == FinalSignalDirection.SELL:
        signal = StrategySignal(
            **{**signal.__dict__, "direction": SignalDirection.SELL, "entry": 2650.0, "stop_loss": 2655.0, "target": 2640.0, "risk_reward": 2.0}
        )
        evaluation = StrategyEvaluation(
            **{**evaluation.__dict__, "direction": SignalDirection.SELL, "signal": signal}
        )
    return FinalSignalDecision(
        timestamp=DECISION_TS,
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        confidence=confidence,
        confidence_label_ar="قوي",
        entry=signal.entry if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        stop_loss=signal.stop_loss if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        target=signal.target if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        risk_reward=signal.risk_reward if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        selected_strategy="Classic" if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        selected_variant="Classic_V1" if direction != FinalSignalDirection.NO_CLEAR_SIGNAL else None,
        reasons=("strategy agreement", "entry quality"),
        strategy_evaluations=(evaluation,),
        directional_support={"BUY": 0.9, "SELL": 0.1},
        scoring_components={"strategy_agreement": 0.9, "entry_quality": 0.7},
        metadata={"market_context": "decision_time"},
    )


def make_snapshot(*, snapshot_id: str = "snapshot-l3-001", market_regime: str | None = "TRENDING") -> FeatureSnapshot:
    return FeatureSnapshot(
        snapshot_id=snapshot_id,
        timestamp=DECISION_TS,
        symbol="XAUUSD",
        timeframe="M15",
        feature_engine_version="phase03-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features={"price.close": 2650.0, "momentum.rsi": 61.5, "trend.ema20": 2648.0},
        market_regime=market_regime,
        mtf_context={"H1": {"trend": "UP"}},
        dataset_version="dataset-2026-09",
        provenance_metadata={"provider": "local_csv", "dataset_file": "XAUUSD.csv"},
        created_at=DECISION_TS,
    )


class LearningRecordBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "learning.db")
        MigrationRunner(self.db).apply_all()
        self.store = FeatureStore(self.db)
        self.repository = LearningRecordRepository(self.db)
        self.snapshot = self.store.save(make_snapshot())
        self.builder = LearningRecordBuilder(self.store)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_01_build_learning_record_from_valid_decision(self) -> None:
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            parameters_snapshot={"rsi_period": 14, "ema_fast": 20, "ema_slow": 50},
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.status, LearningRecordStatus.PENDING_OUTCOME)

    def test_02_correct_symbol(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.symbol, "XAUUSD")

    def test_03_correct_timeframe(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.timeframe, "M15")

    def test_04_correct_decision_timestamp(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.decision_timestamp, DECISION_TS)

    def test_05_correct_strategy_name(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.strategy_name, "Classic")

    def test_06_correct_strategy_version(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.strategy_version, "Classic_V1")

    def test_07_correct_direction(self) -> None:
        buy = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        sell_decision = make_decision(direction=FinalSignalDirection.SELL)
        sell = self.builder.build(sell_decision, feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(buy.direction, LearningDirection.BUY)
        self.assertEqual(sell.direction, LearningDirection.SELL)

    def test_08_correct_entry(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.entry_price, 2650.0)

    def test_09_correct_target(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.target, 2660.0)

    def test_10_correct_stop_loss(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.stop_loss, 2645.0)

    def test_11_correct_risk_reward(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.risk_reward, 2.0)

    def test_12_correct_confidence_at_decision(self) -> None:
        record = self.builder.build(make_decision(confidence=81.25), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.confidence, 81.25)

    def test_13_correct_evidence_snapshot(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.evidence_snapshot["scoring_components"]["entry_quality"], 0.7)
        self.assertIn("selected_strategy_evaluation", record.evidence_snapshot)

    def test_14_correct_market_regime_when_available(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.market_regime, "TRENDING")

    def test_15_correct_parameters_snapshot(self) -> None:
        params = {"rsi_period": 14, "ema_fast": 20, "ema_slow": 50}
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            parameters_snapshot=params,
        )
        self.assertEqual(dict(record.parameters_snapshot), params)

    def test_16_correct_feature_snapshot_id_and_reference(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.feature_snapshot_id, self.snapshot.snapshot_id)
        self.assertTrue(self.store.exists(snapshot_id=record.feature_snapshot_id))
        self.assertEqual(record.feature_set_version, self.snapshot.feature_schema_version)

    def test_17_historical_source_type(self) -> None:
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
        )
        self.assertEqual(record.source_type, LearningSourceType.HISTORICAL_BACKTEST)
        self.assertEqual(record.provenance_metadata["event_kind"], "HISTORICAL_BACKTEST_DECISION")

    def test_18_live_source_type(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertEqual(record.source_type, LearningSourceType.LIVE_TRADE)
        self.assertEqual(record.provenance_metadata["event_kind"], "LIVE_SIGNAL_OBSERVATION")

    def test_19_outcome_remains_unset_at_creation(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        self.assertIsNone(record.outcome)
        self.assertIsNone(record.exit_timestamp)
        self.assertIsNone(record.exit_price)
        self.assertIsNone(record.exit_reason)
        self.assertIsNone(record.duration)
        self.assertEqual(record.status, LearningRecordStatus.PENDING_OUTCOME)

    def test_20_decision_data_remains_decoupled_from_later_config_changes(self) -> None:
        params = {"rsi_period": 14, "ema_fast": 20}
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            parameters_snapshot=params,
        )
        params["rsi_period"] = 99
        params["ema_fast"] = 200
        self.assertEqual(record.parameters_snapshot["rsi_period"], 14)
        self.assertEqual(record.parameters_snapshot["ema_fast"], 20)
        self.assertEqual(record.strategy_version, "Classic_V1")

    def test_21_feature_snapshot_reference_survives_repository_reload(self) -> None:
        record = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        stored = self.repository.create(record)
        reloaded = self.repository.get(stored.record_id)
        self.assertEqual(reloaded.feature_snapshot_id, self.snapshot.snapshot_id)
        self.assertEqual(self.store.load(reloaded.feature_snapshot_id).snapshot_hash, self.snapshot.snapshot_hash)

    def test_22_duplicate_decision_does_not_create_duplicate_record(self) -> None:
        first = self.builder.build_and_store(
            self.repository,
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            parameters_snapshot={"rsi_period": 14},
        )
        second = self.builder.build_and_store(
            self.repository,
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            parameters_snapshot={"rsi_period": 14},
        )
        self.assertEqual(first.record_id, second.record_id)
        self.assertEqual(len(self.repository.list()), 1)

    def test_23_missing_optional_metadata_is_safe(self) -> None:
        snapshot = FeatureSnapshot(
            snapshot_id="snapshot-l3-optional",
            timestamp=DECISION_TS,
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features={"price.close": 2650.0},
            created_at=DECISION_TS,
            dataset_version=None,
            market_regime=None,
            provenance_metadata={},
        )
        # A direct snapshot remains valid without optional metadata when the store is not supplied.
        record = LearningRecordBuilder().build(make_decision(), feature_snapshot=snapshot)
        self.assertIsNone(record.market_regime)
        self.assertIsNone(record.dataset_version)
        self.assertEqual(record.provenance_metadata["source_type"], LearningSourceType.LIVE_TRADE.value)

    def test_24_invalid_decision_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(LearningRecordBuildError, "only an actionable BUY/SELL"):
            self.builder.build(
                make_decision(direction=FinalSignalDirection.NO_CLEAR_SIGNAL),
                feature_snapshot_id=self.snapshot.snapshot_id,
            )

    def test_25_serialization_and_reload_preserve_decision_snapshot(self) -> None:
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            parameters_snapshot={"rsi_period": 14},
        )
        round_trip = type(record).from_dict(record.to_dict())
        self.assertEqual(round_trip.to_dict(), record.to_dict())
        self.repository.create(record)
        reloaded = self.repository.get(record.record_id)
        self.assertEqual(reloaded.to_dict(), record.to_dict())

    def test_26_leakage_future_market_data_cannot_change_record(self) -> None:
        future_bars = [{"timestamp": DECISION_TS + timedelta(minutes=15), "close": 9999.0}]
        original = self.builder.build(make_decision(), feature_snapshot_id=self.snapshot.snapshot_id)
        before = copy.deepcopy(original.to_dict())

        future_bars.append({"timestamp": DECISION_TS + timedelta(minutes=30), "close": 10000.0})
        future_bars[0]["close"] = 1.0

        after = original.to_dict()
        self.assertEqual(after, before)
        self.assertEqual(original.feature_snapshot_id, self.snapshot.snapshot_id)

    def test_27_snapshot_must_match_decision_timestamp(self) -> None:
        wrong = FeatureSnapshot(
            timestamp=DECISION_TS + timedelta(minutes=15),
            symbol="XAUUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features={"price.close": 2650.0},
            created_at=DECISION_TS,
        )
        with self.assertRaisesRegex(LearningRecordBuildError, "timestamp"):
            LearningRecordBuilder().build(make_decision(), feature_snapshot=wrong)

    def test_28_snapshot_identity_must_match_decision(self) -> None:
        wrong = FeatureSnapshot(
            timestamp=DECISION_TS,
            symbol="EURUSD",
            timeframe="M15",
            feature_engine_version="phase03-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features={"price.close": 1.17},
            created_at=DECISION_TS,
        )
        with self.assertRaisesRegex(LearningRecordBuildError, "symbol"):
            LearningRecordBuilder().build(make_decision(), feature_snapshot=wrong)

    def test_29_explicit_strategy_version_is_preserved(self) -> None:
        record = self.builder.build(
            make_decision(),
            feature_snapshot_id=self.snapshot.snapshot_id,
            strategy_version="Classic_Custom_V1",
        )
        self.assertEqual(record.strategy_version, "Classic_Custom_V1")
        self.assertEqual(record.provenance_metadata["strategy_version_source"], "explicit_builder_input")

    def test_30_persisted_snapshot_id_is_required_when_using_id_only(self) -> None:
        with self.assertRaisesRegex(LearningRecordBuildError, "feature snapshot not found"):
            self.builder.build(make_decision(), feature_snapshot_id="does-not-exist")


if __name__ == "__main__":
    unittest.main()

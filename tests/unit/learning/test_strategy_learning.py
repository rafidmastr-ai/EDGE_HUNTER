"""Per-strategy machine learning: features, labels, training gate, inference, live feedback."""

from __future__ import annotations

import json
import random
from collections import Counter
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from app.data.schema import CanonicalOHLC
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.domain.market import OHLCBar
from app.features.engine import FeatureEngine
from app.learning.models import LearningRecordStatus, LearningSourceType
from app.learning.repository import LearningRecordRepository
from app.learning.strategy_learning import (
    FEATURE_NAMES,
    STATUS_ACTIVE,
    STATUS_INSUFFICIENT_DATA,
    STATUS_REJECTED,
    HistoricalSampleBuilder,
    LiveOutcomeResolver,
    LiveSignalRecorder,
    StrategyLearningBundle,
    StrategyLearningError,
    StrategyLearningFilter,
    StrategyLearningTrainer,
    StrategyModel,
    StrategySample,
    base_interval_minutes,
    collect_historical_samples,
    live_samples,
    performance,
    read_ohlc_csv,
    run_training,
    signal_features,
    timeframe_bars,
)
from app.signals.engine import SignalConfidenceEngine
from app.strategies.models import SignalDirection, SignalState, StrategyContext, StrategySignal
from app.strategies.registry import StrategyRegistry

T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def random_walk(count: int, *, seed: int = 7, minutes: int = 1, start: float = 2000.0) -> list[CanonicalOHLC]:
    rng = random.Random(seed)
    bars, price = [], start
    for i in range(count):
        close = price * (1 + rng.gauss(0, 0.0015))
        high = max(price, close) * (1 + abs(rng.gauss(0, 0.0007)))
        low = min(price, close) * (1 - abs(rng.gauss(0, 0.0007)))
        bars.append(
            CanonicalOHLC(
                timestamp=T0 + timedelta(minutes=minutes * i),
                open=Decimal(f"{price:.4f}"),
                high=Decimal(f"{high:.4f}"),
                low=Decimal(f"{low:.4f}"),
                close=Decimal(f"{close:.4f}"),
            )
        )
        price = close
    return bars


def snapshot_values(**overrides) -> dict:
    values = {
        "price.close": 100.0,
        "volatility.atr": 2.0,
        "momentum.rsi": 60.0,
        "momentum.macd": 0.5,
        "momentum.macd_histogram": 0.2,
        "trend.ema_20": 99.0,
        "trend.ema_50": 98.0,
        "trend.ema_200": 95.0,
        "trend.ema_alignment": "BULLISH",
        "trend.ema_spread_20_50_pct": 1.0,
        "structure.range": 2.0,
        "structure.upper_wick": 0.2,
        "structure.lower_wick": 0.4,
        "structure.close_location": 0.8,
        "structure.body_ratio": 0.7,
        "returns.simple_1": 0.002,
        "returns.simple_5": 0.004,
        "returns.simple_20": 0.01,
        "volatility.atr_pct": 2.0,
        "volatility.realized_log_std": 0.002,
    }
    values.update(overrides)
    return values


def make_signal(direction: str = "BUY", *, strategy: str = "Classic", timestamp: datetime = T0, entry: float = 100.0) -> StrategySignal:
    buy = direction == "BUY"
    return StrategySignal(
        timestamp=timestamp,
        symbol="BTC/USD",
        timeframe="M15",
        direction=SignalDirection.BUY if buy else SignalDirection.SELL,
        state=SignalState.SIGNAL,
        entry=entry,
        stop_loss=entry - 2.0 if buy else entry + 2.0,
        target=entry + 3.5 if buy else entry - 3.5,
        risk_reward=1.75,
        entry_logic="test",
        invalidation="test",
        stop_loss_logic="test",
        target_logic="test",
        evidence=("test setup",),
        strategy_name=strategy,
        variant=f"{strategy}_V1",
    )


def context_for(values: dict, timeframe: str = "M15") -> StrategyContext:
    bars = random_walk(260, minutes=15)
    analysis = FeatureEngine().compute(bars, "BTC/USD", timeframe)
    last = analysis.snapshots[-1]
    snapshots = analysis.snapshots[:-1] + (replace(last, values=values),)
    return StrategyContext.from_series(bars, replace(analysis, snapshots=snapshots))


def synthetic_samples(count: int, *, informative: bool, seed: int = 3) -> list[StrategySample]:
    rng = random.Random(seed)
    samples = []
    for i in range(count):
        features = {name: rng.gauss(0, 1) for name in FEATURE_NAMES}
        if informative:
            won = 1 if features["rsi_directional"] + rng.gauss(0, 0.35) > 0.4 else 0
        else:
            won = 1 if rng.random() < 0.35 else 0
        samples.append(
            StrategySample(
                strategy="ICT",
                symbol="XAUUSD",
                timeframe="M15",
                timestamp=T0 + timedelta(minutes=15 * i),
                features=features,
                won=won,
                r_multiple=1.75 if won else -1.0,
                source="HISTORICAL",
            )
        )
    return samples


def active_model(strategy: str = "Classic", *, weight: float = 4.0, threshold: float = 0.5) -> StrategyModel:
    coef = tuple(weight if name == "rsi_directional" else 0.0 for name in FEATURE_NAMES)
    return StrategyModel(
        strategy=strategy,
        status=STATUS_ACTIVE,
        reason="test",
        mean=tuple(0.0 for _ in FEATURE_NAMES),
        scale=tuple(1.0 for _ in FEATURE_NAMES),
        coef=coef,
        intercept=0.0,
        threshold=threshold,
    )


class FeatureTests(unittest.TestCase):
    def test_features_are_direction_aware_and_finite(self) -> None:
        buy = signal_features(make_signal("BUY"), snapshot_values(), "M15")
        sell = signal_features(make_signal("SELL"), snapshot_values(), "M15")
        self.assertEqual(tuple(buy), FEATURE_NAMES)
        self.assertAlmostEqual(buy["rsi_directional"], 0.2)
        self.assertAlmostEqual(sell["rsi_directional"], -0.2)
        self.assertEqual(buy["ema_alignment_directional"], 1.0)
        self.assertEqual(sell["ema_alignment_directional"], -1.0)
        self.assertEqual(buy["timeframe_m15"], 1.0)
        self.assertAlmostEqual(buy["stop_distance_atr"], 1.0)

    def test_missing_indicators_or_non_signal_return_none(self) -> None:
        self.assertIsNone(signal_features(make_signal(), snapshot_values(**{"volatility.atr": None}), "M15"))
        no_signal = StrategySignal(
            timestamp=T0, symbol="X", timeframe="M15", direction=SignalDirection.NO_SIGNAL, state=SignalState.NO_SIGNAL,
            entry=None, stop_loss=None, target=None, risk_reward=None, entry_logic="", invalidation="",
            stop_loss_logic="", target_logic="", strategy_name="Classic", variant="Classic_V1",
        )
        self.assertIsNone(signal_features(no_signal, snapshot_values(), "M15"))


class HistoricalSampleTests(unittest.TestCase):
    def test_windowed_replay_matches_full_history_signals(self) -> None:
        bars = tuple(random_walk(900, seed=11, minutes=5))
        registry = StrategyRegistry.default()
        analysis = FeatureEngine().compute(bars, "XAUUSD", "M5")
        builder = HistoricalSampleBuilder(registry, horizon_bars=10)
        windowed = {(s.strategy, s.timestamp) for s in builder.samples_from_bars("XAUUSD", "M5", bars)}
        full = set()
        for index in range(len(bars) - 1 - 10 + 1):
            if not analysis.snapshots[index].warmup_complete:
                continue
            context = StrategyContext.from_series(bars, analysis, decision_index=index)
            for signal in registry.evaluate_all(context):
                if signal.state == SignalState.SIGNAL and signal_features(signal, context.current.values, "M5") is not None:
                    full.add((signal.strategy_name, signal.timestamp))
        self.assertTrue(full)
        self.assertEqual(windowed, full)

    def test_labels_use_tp_sl_semantics(self) -> None:
        builder = HistoricalSampleBuilder(horizon_bars=5)
        signal = make_signal("BUY")

        def bar(i: int, high: float, low: float) -> CanonicalOHLC:
            return CanonicalOHLC(T0 + timedelta(minutes=15 * i), Decimal("100"), Decimal(str(high)), Decimal(str(low)), Decimal("100"))

        win = builder._label(signal, "BTC/USD", "M15", snapshot_values(), [bar(1, 101, 99), bar(2, 104, 100)])
        loss = builder._label(signal, "BTC/USD", "M15", snapshot_values(), [bar(1, 101, 97.5)])
        expired = builder._label(signal, "BTC/USD", "M15", snapshot_values(), [bar(1, 101, 99.5)])
        self.assertEqual((win.won, win.r_multiple), (1, 1.75))
        self.assertEqual((loss.won, loss.r_multiple), (0, -1.0))
        self.assertEqual(expired.won, 0)
        self.assertAlmostEqual(expired.r_multiple, 0.0)

    def test_cost_is_charged_as_cost_over_risk_in_r(self) -> None:
        builder = HistoricalSampleBuilder(horizon_bars=5, cost_per_trade={"btc/usd": 0.5})
        signal = make_signal("BUY")  # risk = 2.0

        def bar(i: int, high: float, low: float) -> CanonicalOHLC:
            return CanonicalOHLC(T0 + timedelta(minutes=15 * i), Decimal("100"), Decimal(str(high)), Decimal(str(low)), Decimal("100"))

        win = builder._label(signal, "BTC/USD", "M15", snapshot_values(), [bar(1, 104, 100)])
        loss = builder._label(signal, "BTC/USD", "M15", snapshot_values(), [bar(1, 101, 97.5)])
        other = builder._label(signal, "ETH/USD", "M15", snapshot_values(), [bar(1, 104, 100)])
        self.assertEqual(win.won, 1)
        self.assertAlmostEqual(win.r_multiple, 1.75 - 0.25)
        self.assertAlmostEqual(loss.r_multiple, -1.25)
        self.assertAlmostEqual(other.r_multiple, 1.75)

    def test_timeframes_follow_csv_native_interval(self) -> None:
        m1 = random_walk(300)
        h1 = random_walk(300, minutes=60)
        self.assertEqual(base_interval_minutes(m1), 1)
        self.assertEqual(len(timeframe_bars(m1, "M5")), 60)
        self.assertIsNone(timeframe_bars(h1, "M5"))
        self.assertEqual(len(timeframe_bars(h1, "H1")), 300)


class TrainerTests(unittest.TestCase):
    def test_informative_data_produces_active_model_with_oos_gain(self) -> None:
        bundle = StrategyLearningTrainer(min_samples=200).train(synthetic_samples(1500, informative=True))
        model = bundle.models["ICT"]
        self.assertEqual(model.status, STATUS_ACTIVE, model.reason)
        oos = model.metrics["oos"]
        self.assertGreater(oos["filtered_avg_r"], oos["baseline_avg_r"] + 0.02)
        self.assertEqual(bundle.models["Classic"].status, STATUS_INSUFFICIENT_DATA)

    def test_low_base_win_rate_strategy_can_still_be_learned(self) -> None:
        # ~20% winners: model probabilities stay far below 0.5, so thresholds must
        # come from the probability distribution, not from fixed levels.
        rng = random.Random(9)
        samples = []
        for sample in synthetic_samples(3000, informative=False, seed=9):
            won = 1 if sample.features["rsi_directional"] + rng.gauss(0, 0.4) > 1.2 else 0
            samples.append(replace(sample, won=won, r_multiple=1.75 if won else -1.0))
        self.assertLess(sum(item.won for item in samples) / len(samples), 0.25)
        model = StrategyLearningTrainer(min_samples=200).train(samples).models["ICT"]
        self.assertEqual(model.status, STATUS_ACTIVE, model.reason)
        self.assertLess(model.threshold, 0.5)

    def test_noise_is_rejected_and_strategy_stays_unfiltered(self) -> None:
        model = StrategyLearningTrainer(min_samples=200).train(synthetic_samples(1500, informative=False)).models["ICT"]
        self.assertEqual(model.status, STATUS_REJECTED)
        self.assertFalse(model.active)

    def test_too_few_samples_is_insufficient(self) -> None:
        model = StrategyLearningTrainer(min_samples=200).train(synthetic_samples(50, informative=True)).models["ICT"]
        self.assertEqual(model.status, STATUS_INSUFFICIENT_DATA)

    def test_split_is_chronological_so_oos_is_the_latest_segment(self) -> None:
        samples = synthetic_samples(1000, informative=True)
        model = StrategyLearningTrainer(min_samples=200).train(list(reversed(samples))).models["ICT"]
        self.assertEqual(model.metrics["train"], 600)
        self.assertEqual(model.metrics["oos"]["count"], 200)


class PeriodSplitTests(unittest.TestCase):
    def samples_over_years(self) -> list[StrategySample]:
        base = synthetic_samples(3000, informative=True)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        step = (datetime(2025, 9, 1, tzinfo=timezone.utc) - start) / len(base)
        return [replace(item, timestamp=start + step * i, timeframe=("M5", "M15", "H1")[i % 3]) for i, item in enumerate(base)]

    def test_test_period_is_isolated_and_validation_is_end_of_each_year(self) -> None:
        period = (datetime(2023, 11, 14, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc))
        trainer = StrategyLearningTrainer(min_samples=200, test_period=period)
        ordered = sorted(self.samples_over_years(), key=lambda item: item.timestamp)
        train, validation, oos = trainer.split(ordered)
        self.assertEqual(len(train) + len(validation) + len(oos), len(ordered))
        self.assertTrue(all(period[0] <= item.timestamp < period[1] for item in oos))
        self.assertFalse(any(period[0] <= item.timestamp < period[1] for item in [*train, *validation]))
        self.assertEqual({item.timestamp.year for item in validation}, {2020, 2021, 2022, 2023, 2025})
        for year in (2020, 2021, 2022, 2023, 2025):
            train_year = [item.timestamp for item in train if item.timestamp.year == year]
            validation_year = [item.timestamp for item in validation if item.timestamp.year == year]
            self.assertLess(max(train_year), min(validation_year))
            self.assertAlmostEqual(len(validation_year) / (len(train_year) + len(validation_year)), 0.2, delta=0.01)

    def test_training_reports_each_timeframe_per_segment(self) -> None:
        period = (datetime(2023, 11, 14, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc))
        model = StrategyLearningTrainer(min_samples=200, test_period=period).train(self.samples_over_years()).models["ICT"]
        breakdown = model.metrics["by_timeframe"]
        self.assertEqual(list(breakdown["test"]), ["M5", "M15", "H1"])
        self.assertEqual(sum(row["all"]["setups"] for row in breakdown["test"].values()), model.metrics["split"]["test"])
        self.assertIn("filtered", breakdown["test"]["M5"])

    def test_invalid_period_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StrategyLearningTrainer(test_period=(datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc)))

    def test_performance_metrics(self) -> None:
        items = [replace(synthetic_samples(1, informative=False)[0], r_multiple=value, won=int(value > 0), timestamp=T0 + timedelta(hours=i))
                 for i, value in enumerate([1.0, -1.0, -1.0, 2.0])]
        result = performance(items)
        self.assertEqual((result["setups"], result["total_r"], result["max_drawdown_r"], result["profit_factor"]), (4, 1.0, 2.0, 1.5))
        self.assertIsNone(result["avg_r_ci95"])
        self.assertEqual(performance([]), {"setups": 0})


class BundleAndFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "models" / "strategy_learning.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, **models: StrategyModel) -> None:
        StrategyLearningBundle(trained_at=datetime.now(timezone.utc), models=models).save(self.path)

    def test_roundtrip_and_tamper_detection(self) -> None:
        self._save(Classic=active_model())
        loaded = StrategyLearningBundle.load(self.path)
        self.assertTrue(loaded.models["Classic"].active)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        document["payload"]["models"]["Classic"]["threshold"] = 0.01
        self.path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(StrategyLearningError):
            StrategyLearningBundle.load(self.path)
        learning_filter = StrategyLearningFilter(self.path, reload_interval_seconds=0)
        signal = make_signal()
        self.assertEqual(learning_filter.apply((signal,), context_for(snapshot_values(**{"momentum.rsi": 10.0}))), (signal,))
        self.assertEqual(learning_filter.status()["load_error"], "StrategyLearningError")

    def test_low_probability_setup_is_withheld_high_is_annotated(self) -> None:
        self._save(Classic=active_model())
        learning_filter = StrategyLearningFilter(self.path, reload_interval_seconds=0)
        weak = learning_filter.apply((make_signal(),), context_for(snapshot_values(**{"momentum.rsi": 20.0})))[0]
        strong = learning_filter.apply((make_signal(),), context_for(snapshot_values(**{"momentum.rsi": 90.0})))[0]
        self.assertEqual(weak.state, SignalState.NO_SIGNAL)
        self.assertEqual(weak.metadata["original_direction"], "BUY")
        self.assertTrue(weak.evidence[0].startswith("ml_filter_rejected"))
        self.assertEqual(strong.state, SignalState.SIGNAL)
        self.assertGreater(strong.metadata["ml_probability"], 0.5)
        self.assertEqual((strong.entry, strong.stop_loss, strong.target), (100.0, 98.0, 103.5))

    def test_each_strategy_uses_its_own_model_only(self) -> None:
        self._save(SMC=active_model("SMC"))
        learning_filter = StrategyLearningFilter(self.path, reload_interval_seconds=0)
        context = context_for(snapshot_values(**{"momentum.rsi": 20.0}))
        classic, smc, ict = learning_filter.apply(
            (make_signal(strategy="Classic"), make_signal(strategy="SMC"), make_signal(strategy="ICT")), context
        )
        self.assertEqual(classic.state, SignalState.SIGNAL)
        self.assertEqual(smc.state, SignalState.NO_SIGNAL)
        self.assertEqual(ict.state, SignalState.SIGNAL)
        self.assertEqual(learning_filter.active_strategies(), ["SMC"])

    def test_new_training_file_is_picked_up_without_restart(self) -> None:
        learning_filter = StrategyLearningFilter(self.path, reload_interval_seconds=0)
        self.assertFalse(learning_filter.status()["model_loaded"])
        self._save(ICT=active_model("ICT"))
        self.assertEqual(learning_filter.active_strategies(), ["ICT"])

    def test_disabled_filter_changes_nothing(self) -> None:
        self._save(Classic=active_model())
        learning_filter = StrategyLearningFilter(self.path, enabled=False, reload_interval_seconds=0)
        signal = make_signal()
        self.assertEqual(learning_filter.apply((signal,), context_for(snapshot_values(**{"momentum.rsi": 10.0}))), (signal,))

    def test_engine_integration_and_price_guard(self) -> None:
        self._save(Classic=active_model(weight=-50.0))  # withholds every bullish-RSI setup
        context = context_for(snapshot_values(**{"momentum.rsi": 90.0}))
        registry = StrategyRegistry((_FixedStrategy(make_signal()),))
        filtered_engine = SignalConfidenceEngine(registry, signal_filter=StrategyLearningFilter(self.path, reload_interval_seconds=0))
        raw, final = filtered_engine.evaluate_signals(context)
        self.assertEqual(raw[0].state, SignalState.SIGNAL)
        self.assertEqual(final[0].state, SignalState.NO_SIGNAL)
        self.assertEqual(filtered_engine.analyze(context).direction.value, "NO_CLEAR_SIGNAL")
        self.assertEqual(SignalConfidenceEngine(registry).evaluate_signals(context)[1], raw)

        class PriceChangingFilter:
            def apply(self, signals, context):
                return tuple(replace(item, target=item.target + 1) for item in signals)

        with self.assertRaises(ValueError):
            SignalConfidenceEngine(registry, signal_filter=PriceChangingFilter()).evaluate_signals(context)


class _FixedStrategy:
    def __init__(self, signal: StrategySignal) -> None:
        self.signal = signal
        self.name = signal.strategy_name
        self.variant = signal.variant

    def generate(self, context):
        return replace(self.signal, timestamp=context.current.timestamp, timeframe=context.timeframe, symbol=context.symbol)


class FakeProvider:
    name = "fake"

    def __init__(self, bars=None, error: Exception | None = None) -> None:
        self.bars = bars or []
        self.error = error
        self.calls = []

    def get_ohlc(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe))
        if self.error:
            raise self.error
        return list(self.bars)


class LiveFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "edge.db")
        MigrationRunner(self.db).apply_all()
        self.repo = LearningRecordRepository(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def _observation(self, **kwargs):
        signal = make_signal(**kwargs)
        return ("M15", signal, snapshot_values())

    def test_recorder_stores_only_actionable_setups_once(self) -> None:
        recorder = LiveSignalRecorder(self.db)
        observation = self._observation()
        self.assertEqual(recorder.record("BTC/USD", [observation, observation]), 1)
        self.assertEqual(recorder.record("BTC/USD", [observation]), 0)
        records = self.repo.list(source_type=LearningSourceType.LIVE_TRADE)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, LearningRecordStatus.PENDING_OUTCOME)
        self.assertEqual(set(records[0].feature_snapshot.values), set(FEATURE_NAMES))
        self.assertEqual(LiveSignalRecorder(self.db, enabled=False).record("BTC/USD", [self._observation(entry=200.0)]), 0)

    def test_resolver_labels_after_enough_bars_and_waits_otherwise(self) -> None:
        LiveSignalRecorder(self.db).record("BTC/USD", [self._observation()])
        bars = [
            OHLCBar(timestamp=T0 + timedelta(minutes=15 * i), open=Decimal("100"), high=Decimal("101"), low=Decimal("99.5"), close=Decimal("100"))
            for i in range(1, 4)
        ]
        now = T0 + timedelta(days=1)
        waiting = LiveOutcomeResolver(self.db, FakeProvider(bars), horizon_bars=10).resolve(now)
        self.assertEqual((waiting["pending"], waiting["waiting"]), (1, 1))

        bars.append(OHLCBar(timestamp=T0 + timedelta(minutes=60), open=Decimal("100"), high=Decimal("104"), low=Decimal("100"), close=Decimal("103")))
        done = LiveOutcomeResolver(self.db, FakeProvider(bars), horizon_bars=10).resolve(now)
        self.assertEqual(done["completed"], 1)
        samples = live_samples(self.db)
        self.assertEqual(len(samples), 1)
        self.assertEqual((samples[0].won, samples[0].source), (1, "LIVE"))
        self.assertAlmostEqual(samples[0].r_multiple, 1.75)

    def test_resolver_survives_provider_errors(self) -> None:
        LiveSignalRecorder(self.db).record("BTC/USD", [self._observation()])
        summary = LiveOutcomeResolver(self.db, FakeProvider(error=RuntimeError("down"))).resolve()
        self.assertEqual(summary["fetch_errors"], 1)
        self.assertEqual(len(self.repo.list(status=LearningRecordStatus.PENDING_OUTCOME)), 1)


class EndToEndTrainingTests(unittest.TestCase):
    def test_run_training_reads_csv_folder_and_saves_loadable_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "raw"
            data_dir.mkdir()
            with (data_dir / "BTCUSD.csv").open("w", encoding="utf-8") as handle:
                handle.write("timestamp,open,high,low,close\n")
                for bar in random_walk(6000, seed=5):
                    epoch_ms = int(bar.timestamp.timestamp() * 1000)
                    handle.write(f"{epoch_ms},{bar.open},{bar.high},{bar.low},{bar.close}\n")
            self.assertEqual(len(read_ohlc_csv(data_dir / "BTCUSD.csv")), 6000)
            db = Database(root / "edge.db")
            try:
                MigrationRunner(db).apply_all()
                bundle = run_training(
                    data_dir=data_dir,
                    database=db,
                    model_path=root / "models" / "strategy_learning.json",
                    horizon_bars=20,
                    trainer=StrategyLearningTrainer(min_samples=30, min_oos_kept=5),
                )
            finally:
                db.close()
            loaded = StrategyLearningBundle.load(root / "models" / "strategy_learning.json")
            self.assertEqual(set(loaded.models), {"Classic", "SMC", "ICT"})
            self.assertGreater(bundle.data_summary["historical_samples"], 0)
            self.assertEqual(set(bundle.data_summary["files"]["BTCUSD.csv"]["timeframes"]), {"M5", "M15", "H1"})

    def test_mt5_folder_with_test_period_replays_pieces_separately(self) -> None:
        class SpyBuilder(HistoricalSampleBuilder):
            def __init__(self) -> None:
                super().__init__(horizon_bars=20)
                self.series: list[tuple[str, datetime, datetime]] = []

            def samples_from_bars(self, symbol, timeframe, bars):
                self.series.append((timeframe, bars[0].timestamp, bars[-1].timestamp))
                return super().samples_from_bars(symbol, timeframe, bars)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "XAUUSD"
            folder.mkdir()
            (folder / "SOURCE.json").write_text(json.dumps({"symbol": "XAUUSD", "server_tz": "Europe/Athens"}), encoding="utf-8")
            with (folder / "XAUUSD_M1_2026.csv").open("w", encoding="utf-8") as handle:
                handle.write("timestamp,open,high,low,close,tick_volume,spread\n")
                for bar in random_walk(9000, seed=5):
                    handle.write(f"{bar.timestamp.strftime('%Y-%m-%dT%H:%M:%S')}+00:00,{bar.open},{bar.high},{bar.low},{bar.close},1,10\n")
            (Path(tmp) / "IGNORED.csv").write_text("timestamp,open,high,low,close\n", encoding="utf-8")
            builder = SpyBuilder()
            test_end = T0 + timedelta(minutes=7000)
            samples, summary = collect_historical_samples(
                Path(tmp), builder, symbols=["XAUUSD"], timeframes=("M5", "H1"), test_fraction=0.2, test_end=test_end
            )
        start = datetime.fromisoformat(summary["test_period"][0])
        self.assertEqual(datetime.fromisoformat(summary["test_period"][1]), test_end)
        self.assertEqual(summary["files"]["XAUUSD"]["test_share_of_bars"], 0.2)
        self.assertNotIn("IGNORED.csv", summary["files"])
        self.assertEqual(Counter(timeframe for timeframe, _, _ in builder.series), {"M5": 3, "H1": 3})
        for _, first, last in builder.series:
            self.assertTrue(last < start or first >= test_end or (first >= start and last < test_end))
        self.assertTrue(any(start <= item.timestamp < test_end for item in samples))


if __name__ == "__main__":
    unittest.main()

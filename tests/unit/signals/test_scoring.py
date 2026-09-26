from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.signals.config import ConfidenceConfig
from app.signals.scoring import EvidenceOverrides, score_signal_evidence
from app.strategies.models import SignalDirection, SignalState, StrategySignal


def make_signal(
    strategy: str = "Classic",
    variant: str = "Classic_V1",
    rr: float = 1.75,
    metadata=None,
) -> StrategySignal:
    return StrategySignal(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="XAUUSD",
        timeframe="M15",
        direction=SignalDirection.BUY,
        state=SignalState.SIGNAL,
        entry=100.0,
        stop_loss=99.0,
        target=100.0 + rr,
        risk_reward=rr,
        entry_logic="test entry",
        invalidation="test invalidation",
        stop_loss_logic="test stop",
        target_logic="test target",
        evidence=("confirmation",),
        score_inputs={"confirmation": 1.0, "structure": 1.0},
        strategy_name=strategy,
        variant=variant,
        metadata=metadata or {},
    )


class ScoringTests(unittest.TestCase):
    def test_scoring_is_deterministic(self) -> None:
        signal = make_signal()
        first = score_signal_evidence(signal, 1.0)
        second = score_signal_evidence(signal, 1.0)
        self.assertEqual(first, second)

    def test_validated_evidence_changes_only_the_declared_components(self) -> None:
        signal = make_signal()
        baseline_conf, baseline = score_signal_evidence(signal, 1.0)
        improved_conf, improved = score_signal_evidence(
            signal,
            1.0,
            overrides=EvidenceOverrides(oos_performance=1.0, sample_size=1.0, robustness=1.0, market_regime=1.0),
        )
        self.assertGreaterEqual(improved_conf, baseline_conf)
        self.assertEqual(baseline.entry_quality, improved.entry_quality)
        self.assertEqual(baseline.rr_quality, improved.rr_quality)

    def test_rr_quality_is_bounded(self) -> None:
        for rr in (0.5, 1.5, 2.0, 4.0):
            _, evidence = score_signal_evidence(make_signal(rr=rr), 1.0)
            self.assertGreaterEqual(evidence.rr_quality, 0.0)
            self.assertLessEqual(evidence.rr_quality, 1.0)

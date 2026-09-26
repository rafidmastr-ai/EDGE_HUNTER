"""Deterministic Outcome / Label Engine for EDGE HUNTER L4.

This module resolves an already-built L3 LearningRecord against post-entry OHLC
bars. It reuses the Phase 05 validation, TP/SL touch semantics, spread handling,
and intrabar policy without changing the BacktestEngine itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Sequence

from app.backtest.config import BacktestConfig, IntrabarPolicy
from app.backtest.engine import BacktestEngine
from app.backtest.models import TradeOutcome, TradeRecord
from app.data.schema import CanonicalOHLC
from app.learning.models import LearningRecord, LearningRecordStatus, LearningSourceType
from app.learning.repository import LearningRecordRepository, LearningRecordStateError


class LearningOutcome(str, Enum):
    """Stable labels produced by L4."""

    TP_BEFORE_SL = "TP_BEFORE_SL"
    SL_BEFORE_TP = "SL_BEFORE_TP"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"
    CLOSED_OTHER = "CLOSED_OTHER"


class OutcomeResolutionState(str, Enum):
    """Whether an L4 evaluation produced a final persisted label."""

    COMPLETED = "COMPLETED"
    INVALID = "INVALID"
    NOT_ENOUGH_DATA_YET = "NOT_ENOUGH_DATA_YET"


@dataclass(frozen=True)
class OutcomeResolution:
    """Pure result of evaluating one learning record against post-entry OHLC."""

    outcome: LearningOutcome
    state: OutcomeResolutionState
    label_version: str
    source_type: LearningSourceType
    exit_timestamp: datetime | None
    exit_price: float | None
    exit_reason: str | None
    duration: timedelta | None
    bars_held: int = 0
    backtest_outcome: str | None = None


@dataclass(frozen=True)
class _OutcomeTrade:
    """Minimal duck type accepted by the Phase 05 private touch utility."""

    direction: str
    target: float
    stop_loss: float


class OutcomeLabelEngine:
    """Resolve L3 LearningRecords using Phase 05 OHLC outcome semantics."""

    DEFAULT_LABEL_VERSION = "v1"

    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        label_version: str = DEFAULT_LABEL_VERSION,
    ) -> None:
        normalized = label_version.strip()
        if not normalized:
            raise ValueError("label_version must not be empty")
        self.config = config or BacktestConfig()
        self.label_version = normalized
        self._backtest_semantics = BacktestEngine(self.config)

    def evaluate(
        self,
        record: LearningRecord,
        bars: Sequence[CanonicalOHLC],
        *,
        data_complete: bool = False,
    ) -> OutcomeResolution:
        """Evaluate only bars strictly after entry time.

        ``data_complete=False`` means the available series may be a live,
        still-growing observation window. In that case the engine refuses to
        call a not-yet-reached data horizon EXPIRED.
        """
        try:
            self._validate_record(record)
        except ValueError as exc:
            return self._invalid_resolution(record, str(exc))
        records = tuple(bars)
        try:
            self._validate_bars(records)
        except (TypeError, ValueError) as exc:
            return self._invalid_resolution(record, f"INVALID_OHLC: {exc}")

        boundary = record.entry_timestamp or record.decision_timestamp
        future = tuple(bar for bar in records if bar.timestamp > boundary)
        if not future:
            return OutcomeResolution(
                outcome=LearningOutcome.INVALID,
                state=OutcomeResolutionState.NOT_ENOUGH_DATA_YET,
                label_version=self.label_version,
                source_type=record.source_type,
                exit_timestamp=None,
                exit_price=None,
                exit_reason=None,
                duration=None,
                bars_held=0,
            )

        trade = _OutcomeTrade(
            direction=record.direction.value,
            target=float(record.target),
            stop_loss=float(record.stop_loss),
        )

        for bars_held, bar in enumerate(future, start=1):
            tp_touched, sl_touched = BacktestEngine._touches(trade, bar)
            if tp_touched and sl_touched:
                if self.config.intrabar_policy == IntrabarPolicy.SKIP_TRADE:
                    return self._ambiguous_result(record, bar, bars_held)
                if self.config.intrabar_policy == IntrabarPolicy.TARGET_FIRST:
                    return self._closed_result(
                        record,
                        LearningOutcome.TP_BEFORE_SL,
                        bar,
                        bars_held,
                        float(record.target),
                        "TP_AND_SL_TARGET_FIRST",
                    )
                return self._closed_result(
                    record,
                    LearningOutcome.SL_BEFORE_TP,
                    bar,
                    bars_held,
                    float(record.stop_loss),
                    "TP_AND_SL_STOP_FIRST",
                )

            if tp_touched:
                return self._closed_result(
                    record,
                    LearningOutcome.TP_BEFORE_SL,
                    bar,
                    bars_held,
                    float(record.target),
                    "TP",
                )
            if sl_touched:
                return self._closed_result(
                    record,
                    LearningOutcome.SL_BEFORE_TP,
                    bar,
                    bars_held,
                    float(record.stop_loss),
                    "SL",
                )

            if self.config.max_bars_in_trade is not None and bars_held >= self.config.max_bars_in_trade:
                return self._closed_result(
                    record,
                    LearningOutcome.EXPIRED,
                    bar,
                    bars_held,
                    float(bar.close),
                    "EXPIRY",
                )

        if not data_complete:
            return OutcomeResolution(
                outcome=LearningOutcome.INVALID,
                state=OutcomeResolutionState.NOT_ENOUGH_DATA_YET,
                label_version=self.label_version,
                source_type=record.source_type,
                exit_timestamp=None,
                exit_price=None,
                exit_reason=None,
                duration=None,
                bars_held=len(future),
            )

        # Phase 05 explicitly records data-end as EXPIRED once the historical
        # input is known to be complete. The entry bar itself is never used as
        # an exit observation here; only post-entry bars can be a data-end bar.
        last = future[-1]
        return self._closed_result(
            record,
            LearningOutcome.EXPIRED,
            last,
            len(future),
            float(last.close),
            "DATA_END",
        )

    def resolve_and_store(
        self,
        repository: LearningRecordRepository,
        record_id: str,
        bars: Sequence[CanonicalOHLC],
        *,
        data_complete: bool = False,
    ) -> LearningRecord:
        """Resolve one pending record and persist the result idempotently."""
        current = repository.get(record_id)
        if current is None:
            raise KeyError(record_id)
        if current.status in {LearningRecordStatus.COMPLETED, LearningRecordStatus.INVALID}:
            return current
        if current.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot resolve {current.status.value} record")

        resolution = self.evaluate(current, bars, data_complete=data_complete)
        if resolution.state == OutcomeResolutionState.NOT_ENOUGH_DATA_YET:
            return current
        if resolution.state == OutcomeResolutionState.COMPLETED:
            assert resolution.exit_timestamp is not None
            assert resolution.exit_price is not None
            assert resolution.exit_reason is not None
            assert resolution.duration is not None
            return repository.mark_completed(
                record_id,
                outcome=resolution.outcome.value,
                exit_timestamp=resolution.exit_timestamp,
                exit_price=resolution.exit_price,
                exit_reason=resolution.exit_reason,
                duration=resolution.duration,
                label_version=resolution.label_version,
            )

        return repository.mark_invalid(
            record_id,
            label_version=resolution.label_version,
            outcome=resolution.outcome.value,
            exit_timestamp=resolution.exit_timestamp,
            exit_price=resolution.exit_price,
            exit_reason=resolution.exit_reason or "INVALID_OUTCOME",
            duration=resolution.duration,
        )

    def resolve_historical_trade(
        self,
        repository: LearningRecordRepository,
        record_id: str,
        trade: TradeRecord,
    ) -> LearningRecord:
        """Attach a Phase 05 TradeRecord outcome without recomputing the backtest."""
        current = repository.get(record_id)
        if current is None:
            raise KeyError(record_id)
        if current.status in {LearningRecordStatus.COMPLETED, LearningRecordStatus.INVALID}:
            return current
        if current.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot resolve {current.status.value} record")
        if current.source_type != LearningSourceType.HISTORICAL_BACKTEST:
            raise ValueError("historical TradeRecord adapter requires HISTORICAL_BACKTEST source")
        self._validate_trade_alignment(current, trade)

        resolution = self.from_trade_record(trade, label_version=self.label_version)
        if resolution.state == OutcomeResolutionState.COMPLETED:
            assert resolution.exit_timestamp is not None
            assert resolution.exit_price is not None
            assert resolution.exit_reason is not None
            assert resolution.duration is not None
            return repository.mark_completed(
                record_id,
                outcome=resolution.outcome.value,
                exit_timestamp=resolution.exit_timestamp,
                exit_price=resolution.exit_price,
                exit_reason=resolution.exit_reason,
                duration=resolution.duration,
                label_version=resolution.label_version,
            )
        return repository.mark_invalid(
            record_id,
            label_version=resolution.label_version,
            outcome=resolution.outcome.value,
            exit_timestamp=resolution.exit_timestamp,
            exit_price=resolution.exit_price,
            exit_reason=resolution.exit_reason or "INVALID_TRADE_OUTCOME",
            duration=resolution.duration,
        )

    @classmethod
    def from_trade_record(
        cls,
        trade: TradeRecord,
        *,
        label_version: str = DEFAULT_LABEL_VERSION,
    ) -> OutcomeResolution:
        """Map Phase 05 outcomes into stable learning labels."""
        if not isinstance(trade, TradeRecord):
            raise TypeError("trade must be a TradeRecord")
        normalized = label_version.strip()
        if not normalized:
            raise ValueError("label_version must not be empty")
        duration = trade.exit_timestamp - trade.entry_timestamp
        if duration.total_seconds() < 0:
            raise ValueError("TradeRecord exit_timestamp must not precede entry_timestamp")

        if trade.outcome == TradeOutcome.WIN:
            label = LearningOutcome.TP_BEFORE_SL
            state = OutcomeResolutionState.COMPLETED
        elif trade.outcome == TradeOutcome.LOSS:
            label = LearningOutcome.SL_BEFORE_TP
            state = OutcomeResolutionState.COMPLETED
        elif trade.outcome == TradeOutcome.EXPIRED:
            label = LearningOutcome.EXPIRED
            state = OutcomeResolutionState.COMPLETED
        elif trade.outcome == TradeOutcome.AMBIGUOUS_SKIPPED:
            label = LearningOutcome.INVALID
            state = OutcomeResolutionState.INVALID
        else:  # Defensive future-proofing if Phase 05 gains a new outcome.
            label = LearningOutcome.CLOSED_OTHER
            state = OutcomeResolutionState.COMPLETED

        return OutcomeResolution(
            outcome=label,
            state=state,
            label_version=normalized,
            source_type=LearningSourceType.HISTORICAL_BACKTEST,
            exit_timestamp=trade.exit_timestamp,
            exit_price=float(trade.exit_price),
            exit_reason=trade.exit_reason,
            duration=duration,
            bars_held=max(0, int(trade.bars_held)),
            backtest_outcome=trade.outcome.value,
        )

    def _invalid_resolution(self, record: LearningRecord, reason: str) -> OutcomeResolution:
        return OutcomeResolution(
            outcome=LearningOutcome.INVALID,
            state=OutcomeResolutionState.INVALID,
            label_version=self.label_version,
            source_type=record.source_type,
            exit_timestamp=None,
            exit_price=None,
            exit_reason=reason,
            duration=None,
            bars_held=0,
        )

    def _closed_result(
        self,
        record: LearningRecord,
        outcome: LearningOutcome,
        bar: CanonicalOHLC,
        bars_held: int,
        requested_exit_price: float,
        reason: str,
    ) -> OutcomeResolution:
        exit_price = self._backtest_semantics._effective_exit_price(record.direction.value, requested_exit_price)
        duration = bar.timestamp - (record.entry_timestamp or record.decision_timestamp)
        return OutcomeResolution(
            outcome=outcome,
            state=OutcomeResolutionState.COMPLETED,
            label_version=self.label_version,
            source_type=record.source_type,
            exit_timestamp=bar.timestamp,
            exit_price=float(exit_price),
            exit_reason=reason,
            duration=duration,
            bars_held=bars_held,
        )

    def _ambiguous_result(
        self,
        record: LearningRecord,
        bar: CanonicalOHLC,
        bars_held: int,
    ) -> OutcomeResolution:
        duration = bar.timestamp - (record.entry_timestamp or record.decision_timestamp)
        return OutcomeResolution(
            outcome=LearningOutcome.INVALID,
            state=OutcomeResolutionState.INVALID,
            label_version=self.label_version,
            source_type=record.source_type,
            exit_timestamp=bar.timestamp,
            exit_price=float(bar.close),
            exit_reason="TP_AND_SL_SKIP_TRADE",
            duration=duration,
            bars_held=bars_held,
            backtest_outcome=TradeOutcome.AMBIGUOUS_SKIPPED.value,
        )

    @staticmethod
    def _validate_record(record: LearningRecord) -> None:
        if not isinstance(record, LearningRecord):
            raise TypeError("record must be a LearningRecord")
        if record.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot evaluate {record.status.value} record")
        if record.entry_price <= 0 or record.target <= 0 or record.stop_loss <= 0:
            raise ValueError("learning record contains invalid trade prices")
        if record.direction.value == "BUY":
            valid = record.stop_loss < record.entry_price < record.target
        elif record.direction.value == "SELL":
            valid = record.target < record.entry_price < record.stop_loss
        else:
            valid = False
        if not valid:
            raise ValueError("learning record price geometry is invalid for its direction")

    @staticmethod
    def _validate_bars(records: Sequence[CanonicalOHLC]) -> None:
        if not records:
            return
        BacktestEngine._validate_bars(records)

    @staticmethod
    def _validate_trade_alignment(record: LearningRecord, trade: TradeRecord) -> None:
        checks = {
            "symbol": (record.symbol.upper(), trade.symbol.upper()),
            "timeframe": (record.timeframe, trade.timeframe),
            "direction": (record.direction.value, trade.direction),
        }
        for field, (expected, actual) in checks.items():
            if expected != actual:
                raise ValueError(f"historical TradeRecord {field} does not match learning record")
        if trade.entry_timestamp < (record.entry_timestamp or record.decision_timestamp):
            raise ValueError("historical TradeRecord entry precedes learning record entry boundary")
        if abs(float(trade.stop_loss) - float(record.stop_loss)) > 1e-12:
            raise ValueError("historical TradeRecord stop_loss does not match learning record")
        if abs(float(trade.target) - float(record.target)) > 1e-12:
            raise ValueError("historical TradeRecord target does not match learning record")


__all__ = [
    "LearningOutcome",
    "OutcomeLabelEngine",
    "OutcomeResolution",
    "OutcomeResolutionState",
]

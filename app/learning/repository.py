"""SQLite persistence for the EDGE HUNTER Learning Foundation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db.database import Database
from app.learning.models import (
    LearningDirection,
    LearningFeatureSnapshot,
    LearningRecord,
    LearningRecordStatus,
    LearningSourceType,
)


class DuplicateLearningRecordError(ValueError):
    """Raised when one record id conflicts with a different record identity."""


class LearningRecordStateError(ValueError):
    """Raised when a lifecycle transition is not permitted."""


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _identity_hash(record: LearningRecord) -> str:
    payload = record.canonical_identity_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LearningRecordRepository:
    """Small deterministic repository; no training behavior belongs here."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, record: LearningRecord) -> LearningRecord:
        """Create a record or return the exact existing identity idempotently."""
        identity_hash = _identity_hash(record)
        existing_by_id = self.get(record.record_id)
        if existing_by_id is not None:
            if _identity_hash(existing_by_id) != identity_hash:
                raise DuplicateLearningRecordError(
                    f"record_id {record.record_id!r} already belongs to a different learning record"
                )
            return existing_by_id

        existing_by_identity = self._get_by_identity_hash(identity_hash)
        if existing_by_identity is not None:
            return existing_by_identity

        self.database.execute(
            """
            INSERT INTO learning_records(
                record_id, identity_hash, source_type, symbol, timeframe,
                decision_timestamp, entry_timestamp, direction, entry_price,
                target, stop_loss, risk_reward, strategy_name, strategy_variant,
                strategy_version, feature_set_version, feature_snapshot_id, parameters_snapshot_json,
                feature_snapshot_json, evidence_snapshot_json, confidence,
                market_regime, outcome, exit_timestamp, exit_price, exit_reason,
                duration_seconds, dataset_version, label_version, provenance_metadata_json,
                status, created_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                identity_hash,
                record.source_type.value,
                record.symbol.upper(),
                record.timeframe,
                record.decision_timestamp.astimezone(timezone.utc).isoformat(),
                None if record.entry_timestamp is None else record.entry_timestamp.astimezone(timezone.utc).isoformat(),
                record.direction.value,
                float(record.entry_price),
                float(record.target),
                float(record.stop_loss),
                float(record.risk_reward),
                record.strategy_name,
                record.strategy_variant,
                record.strategy_version,
                record.feature_set_version,
                record.feature_snapshot_id,
                _json(dict(record.parameters_snapshot)),
                _json(record.feature_snapshot.to_dict()),
                _json(dict(record.evidence_snapshot)),
                float(record.confidence),
                record.market_regime,
                record.outcome,
                None if record.exit_timestamp is None else record.exit_timestamp.astimezone(timezone.utc).isoformat(),
                record.exit_price,
                record.exit_reason,
                None if record.duration is None else int(record.duration.total_seconds()),
                record.dataset_version,
                record.label_version,
                _json(dict(record.provenance_metadata)),
                record.status.value,
                record.created_at.astimezone(timezone.utc).isoformat(),
            ),
        )
        self.database.commit()
        return self.get(record.record_id)  # type: ignore[return-value]

    def get(self, record_id: str) -> LearningRecord | None:
        """Return one learning record by id."""
        row = self.database.execute(
            "SELECT * FROM learning_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    def list(
        self,
        *,
        status: LearningRecordStatus | None = None,
        source_type: LearningSourceType | None = None,
        symbol: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LearningRecord]:
        """Return records in deterministic creation order, newest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type.value)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        rows = self.database.execute(
            f"SELECT * FROM learning_records{where} ORDER BY created_at DESC, record_id DESC LIMIT ? OFFSET ?",
            tuple(params + [safe_limit, safe_offset]),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def check_duplicate(self, record: LearningRecord) -> bool:
        """Return True if the record identity or record id already exists."""
        return self.get(record.record_id) is not None or self._get_by_identity_hash(_identity_hash(record)) is not None

    def update_status(self, record_id: str, status: LearningRecordStatus) -> LearningRecord:
        """Perform a safe lifecycle transition that does not fabricate an outcome."""
        current = self.get(record_id)
        if current is None:
            raise KeyError(record_id)
        if current.status == status:
            return current
        if current.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot transition {current.status.value} -> {status.value}")
        if status == LearningRecordStatus.COMPLETED:
            raise LearningRecordStateError("use mark_completed() to attach an outcome")
        if status not in {LearningRecordStatus.INVALID, LearningRecordStatus.EXCLUDED}:
            raise LearningRecordStateError(f"unsupported lifecycle transition to {status.value}")
        self.database.execute(
            "UPDATE learning_records SET status = ? WHERE record_id = ?",
            (status.value, record_id),
        )
        self.database.commit()
        return self.get(record_id)  # type: ignore[return-value]

    def mark_completed(
        self,
        record_id: str,
        *,
        outcome: str,
        exit_timestamp: datetime,
        exit_price: float,
        exit_reason: str,
        duration: timedelta,
        label_version: str = "v1",
    ) -> LearningRecord:
        """Attach a complete outcome and move the record to COMPLETED."""
        current = self.get(record_id)
        if current is None:
            raise KeyError(record_id)
        candidate = LearningRecord(
            record_id=current.record_id,
            source_type=current.source_type,
            symbol=current.symbol,
            timeframe=current.timeframe,
            decision_timestamp=current.decision_timestamp,
            entry_timestamp=current.entry_timestamp,
            direction=current.direction,
            entry_price=current.entry_price,
            target=current.target,
            stop_loss=current.stop_loss,
            risk_reward=current.risk_reward,
            strategy_name=current.strategy_name,
            strategy_variant=current.strategy_variant,
            strategy_version=current.strategy_version,
            feature_set_version=current.feature_set_version,
            feature_snapshot_id=current.feature_snapshot_id,
            parameters_snapshot=current.parameters_snapshot,
            feature_snapshot=current.feature_snapshot,
            evidence_snapshot=current.evidence_snapshot,
            confidence=current.confidence,
            market_regime=current.market_regime,
            outcome=outcome,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            duration=duration,
            dataset_version=current.dataset_version,
            label_version=label_version,
            provenance_metadata=current.provenance_metadata,
            status=LearningRecordStatus.COMPLETED,
            created_at=current.created_at,
        )
        if current.status == LearningRecordStatus.COMPLETED:
            if current.to_dict() != candidate.to_dict():
                raise LearningRecordStateError("COMPLETED record cannot be overwritten with a different outcome")
            return current
        if current.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot complete {current.status.value} record")

        self.database.execute(
            """
            UPDATE learning_records
            SET outcome = ?, exit_timestamp = ?, exit_price = ?, exit_reason = ?,
                duration_seconds = ?, label_version = ?, status = 'COMPLETED'
            WHERE record_id = ? AND status = 'PENDING_OUTCOME'
            """,
            (
                candidate.outcome,
                candidate.exit_timestamp.astimezone(timezone.utc).isoformat(),
                candidate.exit_price,
                candidate.exit_reason,
                int(candidate.duration.total_seconds()),
                candidate.label_version,
                record_id,
            ),
        )
        self.database.commit()
        return self.get(record_id)  # type: ignore[return-value]

    def mark_invalid(
        self,
        record_id: str,
        *,
        label_version: str = "v1",
        outcome: str = "INVALID",
        exit_timestamp: datetime | None = None,
        exit_price: float | None = None,
        exit_reason: str = "INVALID_OUTCOME",
        duration: timedelta | None = None,
    ) -> LearningRecord:
        """Attach a terminal INVALID outcome without changing decision-time data."""
        current = self.get(record_id)
        if current is None:
            raise KeyError(record_id)
        normalized_label = label_version.strip()
        if not normalized_label or not outcome.strip() or not exit_reason.strip():
            raise ValueError("label_version, outcome and exit_reason must not be empty")
        candidate = LearningRecord(
            record_id=current.record_id,
            source_type=current.source_type,
            symbol=current.symbol,
            timeframe=current.timeframe,
            decision_timestamp=current.decision_timestamp,
            entry_timestamp=current.entry_timestamp,
            direction=current.direction,
            entry_price=current.entry_price,
            target=current.target,
            stop_loss=current.stop_loss,
            risk_reward=current.risk_reward,
            strategy_name=current.strategy_name,
            strategy_variant=current.strategy_variant,
            strategy_version=current.strategy_version,
            feature_set_version=current.feature_set_version,
            feature_snapshot_id=current.feature_snapshot_id,
            parameters_snapshot=current.parameters_snapshot,
            feature_snapshot=current.feature_snapshot,
            evidence_snapshot=current.evidence_snapshot,
            confidence=current.confidence,
            market_regime=current.market_regime,
            outcome=outcome,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            duration=duration,
            dataset_version=current.dataset_version,
            label_version=normalized_label,
            provenance_metadata=current.provenance_metadata,
            status=LearningRecordStatus.INVALID,
            created_at=current.created_at,
        )
        if current.status == LearningRecordStatus.INVALID:
            if current.to_dict() != candidate.to_dict():
                raise LearningRecordStateError("INVALID record cannot be overwritten with a different outcome")
            return current
        if current.status != LearningRecordStatus.PENDING_OUTCOME:
            raise LearningRecordStateError(f"cannot invalidate {current.status.value} record")

        self.database.execute(
            """
            UPDATE learning_records
            SET outcome = ?, exit_timestamp = ?, exit_price = ?, exit_reason = ?,
                duration_seconds = ?, label_version = ?, status = 'INVALID'
            WHERE record_id = ? AND status = 'PENDING_OUTCOME'
            """,
            (
                candidate.outcome,
                None if candidate.exit_timestamp is None else candidate.exit_timestamp.astimezone(timezone.utc).isoformat(),
                candidate.exit_price,
                candidate.exit_reason,
                None if candidate.duration is None else int(candidate.duration.total_seconds()),
                candidate.label_version,
                record_id,
            ),
        )
        self.database.commit()
        return self.get(record_id)  # type: ignore[return-value]

    def _get_by_identity_hash(self, identity_hash: str) -> LearningRecord | None:
        row = self.database.execute(
            "SELECT * FROM learning_records WHERE identity_hash = ?",
            (identity_hash,),
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: Any) -> LearningRecord:
        duration_seconds = row["duration_seconds"]
        return LearningRecord(
            record_id=row["record_id"],
            source_type=LearningSourceType(row["source_type"]),
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            decision_timestamp=_dt(row["decision_timestamp"]),  # type: ignore[arg-type]
            entry_timestamp=_dt(row["entry_timestamp"]),
            direction=LearningDirection(row["direction"]),
            entry_price=float(row["entry_price"]),
            target=float(row["target"]),
            stop_loss=float(row["stop_loss"]),
            risk_reward=float(row["risk_reward"]),
            strategy_name=row["strategy_name"],
            strategy_variant=row["strategy_variant"],
            strategy_version=row["strategy_version"],
            feature_set_version=row["feature_set_version"],
            feature_snapshot_id=row["feature_snapshot_id"],
            parameters_snapshot=json.loads(row["parameters_snapshot_json"]),
            feature_snapshot=LearningFeatureSnapshot(json.loads(row["feature_snapshot_json"])),
            evidence_snapshot=json.loads(row["evidence_snapshot_json"]),
            confidence=float(row["confidence"]),
            market_regime=row["market_regime"],
            outcome=row["outcome"],
            exit_timestamp=_dt(row["exit_timestamp"]),
            exit_price=None if row["exit_price"] is None else float(row["exit_price"]),
            exit_reason=row["exit_reason"],
            duration=None if duration_seconds is None else timedelta(seconds=int(duration_seconds)),
            dataset_version=row["dataset_version"],
            label_version=row["label_version"],
            provenance_metadata=json.loads(row["provenance_metadata_json"]),
            status=LearningRecordStatus(row["status"]),
            created_at=_dt(row["created_at"]),  # type: ignore[arg-type]
        )

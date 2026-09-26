"""SQLite persistence for L10 learning-cycle, candidate, production, monitoring and audit state."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from app.db.database import Database


class L10RepositoryError(RuntimeError):
    pass


class LearningCycleStatus(str, Enum):
    DATA_COLLECTION = "DATA_COLLECTION"
    DATA_COMPLETION = "DATA_COMPLETION"
    DATASET_BUILD = "DATASET_BUILD"
    TRAINING = "TRAINING"
    REGISTRATION = "REGISTRATION"
    VALIDATION = "VALIDATION"
    OOS = "OOS"
    ROBUSTNESS = "ROBUSTNESS"
    CANDIDATE = "CANDIDATE"
    PROMOTION_GATE = "PROMOTION_GATE"
    PROMOTED = "PROMOTED"
    MONITORING = "MONITORING"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class CandidateStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    CHALLENGER = "CHALLENGER"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class LearningCycle:
    cycle_id: str
    identity_hash: str
    scope_key: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    trigger: str
    trigger_key: str
    status: LearningCycleStatus
    started_at: datetime
    completed_at: datetime | None
    dataset_version_id: str | None
    training_run_id: str | None
    model_version_id: str | None
    policy_version_id: str | None
    validation_evaluation_id: str | None
    oos_evaluation_id: str | None
    robustness_evaluation_id: str | None
    configuration_fingerprint: str
    retry_count: int = 0
    result_summary: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def create(
        cls,
        *,
        identity_hash: str,
        scope_key: str,
        strategy_name: str,
        strategy_variant: str,
        strategy_version: str,
        trigger: Enum,
        trigger_key: str,
        configuration_fingerprint: str,
        started_at: datetime,
    ) -> "LearningCycle":
        cycle_id = f"cyc-{identity_hash[:32]}"
        return cls(
            cycle_id=cycle_id,
            identity_hash=identity_hash,
            scope_key=scope_key,
            strategy_name=strategy_name,
            strategy_variant=strategy_variant,
            strategy_version=strategy_version,
            trigger=trigger.value,
            trigger_key=trigger_key,
            status=LearningCycleStatus.DATA_COLLECTION,
            started_at=started_at,
            completed_at=None,
            dataset_version_id=None,
            training_run_id=None,
            model_version_id=None,
            policy_version_id=None,
            validation_evaluation_id=None,
            oos_evaluation_id=None,
            robustness_evaluation_id=None,
            configuration_fingerprint=configuration_fingerprint,
        )


@dataclass(frozen=True)
class LearningState:
    scope_key: str
    last_successful_cycle_id: str | None
    last_dataset_version_id: str | None
    last_dataset_end: datetime | None
    last_training_run_id: str | None
    last_candidate_id: str | None
    active_model_version_id: str | None
    active_policy_version_id: str | None
    last_failure: str | None
    last_trigger: str | None
    updated_at: datetime


@dataclass(frozen=True)
class ProductionModelState:
    scope_key: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    model_version_id: str | None
    policy_version_id: str | None
    activated_at: datetime
    activation_reason: str
    source_cycle_id: str | None
    previous_model_version_id: str | None
    previous_policy_version_id: str | None
    status: str


@dataclass(frozen=True)
class MonitoringSnapshot:
    snapshot_id: str
    scope_key: str
    model_version_id: str | None
    policy_version_id: str | None
    observed_at: datetime
    sample_count: int
    metrics: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any]
    drift_score: float
    drift_detected: bool
    runtime_error_count: int
    artifact_healthy: bool

    @classmethod
    def create(cls, **kwargs: Any) -> "MonitoringSnapshot":
        payload = {k: kwargs[k] for k in sorted(kwargs)}
        identity = _sha256({
            "scope_key": kwargs["scope_key"],
            "model_version_id": kwargs["model_version_id"],
            "policy_version_id": kwargs["policy_version_id"],
            "observed_at": _iso(kwargs["observed_at"]),
            "sample_count": kwargs["sample_count"],
            "metrics": kwargs["metrics"],
            "baseline_metrics": kwargs["baseline_metrics"],
            "drift_score": kwargs["drift_score"],
            "runtime_error_count": kwargs["runtime_error_count"],
            "artifact_healthy": kwargs["artifact_healthy"],
        })
        return cls(snapshot_id=f"mon-{identity[:32]}", **kwargs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MonitoringSnapshot":
        return cls(
            snapshot_id=str(value["snapshot_id"]),
            scope_key=str(value["scope_key"]),
            model_version_id=value.get("model_version_id"),
            policy_version_id=value.get("policy_version_id"),
            observed_at=_dt(value["observed_at"]),
            sample_count=int(value["sample_count"]),
            metrics=dict(value.get("metrics", {})),
            baseline_metrics=dict(value.get("baseline_metrics", {})),
            drift_score=float(value.get("drift_score", 0.0)),
            drift_detected=bool(value.get("drift_detected", False)),
            runtime_error_count=int(value.get("runtime_error_count", 0)),
            artifact_healthy=bool(value.get("artifact_healthy", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "scope_key": self.scope_key,
            "model_version_id": self.model_version_id,
            "policy_version_id": self.policy_version_id,
            "observed_at": _iso(self.observed_at),
            "sample_count": self.sample_count,
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "runtime_error_count": self.runtime_error_count,
            "artifact_healthy": self.artifact_healthy,
        }


@dataclass(frozen=True)
class PromotionHistoryEntry:
    history_id: str
    event_type: str
    scope_key: str
    cycle_id: str | None
    old_model_version_id: str | None
    new_model_version_id: str | None
    old_policy_version_id: str | None
    new_policy_version_id: str | None
    reason: str
    policy_version: str | None
    evidence: Mapping[str, Any]
    status: str
    created_at: datetime


@dataclass(frozen=True)
class LearningCandidate:
    candidate_id: str
    identity_hash: str
    cycle_id: str
    scope_key: str
    strategy_name: str
    strategy_variant: str
    strategy_version: str
    model_version_id: str | None
    policy_version_id: str
    dataset_version_id: str
    validation_evaluation_id: str | None
    oos_evaluation_id: str | None
    robustness_evaluation_id: str | None
    status: CandidateStatus
    gate_eligible: bool
    gate_reasons: tuple[str, ...]
    created_at: datetime
    evidence: Mapping[str, Any] = field(default_factory=dict)


class LearningCycleRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, cycle: LearningCycle) -> LearningCycle:
        existing = self.get(cycle.cycle_id)
        if existing is not None:
            if existing.identity_hash != cycle.identity_hash:
                raise L10RepositoryError("cycle identity collision")
            return existing
        try:
            self.database.execute(
                """INSERT INTO learning_cycles(
                    cycle_id, identity_hash, scope_key, strategy_name, strategy_variant, strategy_version,
                    trigger_type, trigger_key, status, started_at, completed_at, dataset_version_id,
                    training_run_id, model_version_id, policy_version_id, validation_evaluation_id,
                    oos_evaluation_id, robustness_evaluation_id, configuration_fingerprint, retry_count,
                    result_summary_json, error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cycle.cycle_id, cycle.identity_hash, cycle.scope_key, cycle.strategy_name,
                    cycle.strategy_variant, cycle.strategy_version, cycle.trigger, cycle.trigger_key,
                    cycle.status.value, _iso(cycle.started_at), _iso(cycle.completed_at), cycle.dataset_version_id,
                    cycle.training_run_id, cycle.model_version_id, cycle.policy_version_id,
                    cycle.validation_evaluation_id, cycle.oos_evaluation_id, cycle.robustness_evaluation_id,
                    cycle.configuration_fingerprint, cycle.retry_count, _json(cycle.result_summary),
                    cycle.error_type, cycle.error_message,
                ),
            )
            self.database.commit()
        except sqlite3.IntegrityError as exc:
            # The partial unique index protects against concurrent active cycles for a scope.
            # Concurrent callers must observe the already-created logical cycle instead of
            # surfacing a race-condition error.
            existing = self.get_by_identity(cycle.identity_hash)
            if existing is not None:
                return existing
            raise L10RepositoryError("active learning cycle already exists for this scope") from exc
        return self.get(cycle.cycle_id) or cycle

    def get(self, cycle_id: str) -> LearningCycle | None:
        row = self.database.execute("SELECT * FROM learning_cycles WHERE cycle_id = ?", (cycle_id,)).fetchone()
        return None if row is None else _cycle_from_row(row)


    def list_cycles(self, *, scope_key: str | None = None) -> tuple[LearningCycle, ...]:
        if scope_key is None:
            rows = self.database.execute("SELECT * FROM learning_cycles ORDER BY started_at ASC, cycle_id ASC").fetchall()
        else:
            rows = self.database.execute("SELECT * FROM learning_cycles WHERE scope_key=? ORDER BY started_at ASC, cycle_id ASC", (scope_key,)).fetchall()
        return tuple(_cycle_from_row(row) for row in rows)

    def get_by_identity(self, identity_hash: str) -> LearningCycle | None:
        row = self.database.execute("SELECT * FROM learning_cycles WHERE identity_hash = ?", (identity_hash,)).fetchone()
        return None if row is None else _cycle_from_row(row)

    def set_status(self, cycle_id: str, status: LearningCycleStatus) -> LearningCycle:
        current = self.get(cycle_id)
        if current is None:
            raise KeyError(cycle_id)
        self.database.execute("UPDATE learning_cycles SET status = ? WHERE cycle_id = ?", (status.value, cycle_id))
        self.database.commit()
        return self.get(cycle_id)  # type: ignore[return-value]

    def mark_retry(self, cycle_id: str, count: int) -> LearningCycle:
        self.database.execute("UPDATE learning_cycles SET retry_count = ? WHERE cycle_id = ?", (count, cycle_id))
        self.database.commit()
        return self.get(cycle_id)  # type: ignore[return-value]

    def complete_success(
        self,
        cycle_id: str,
        status: LearningCycleStatus,
        *,
        candidate_id: str | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> LearningCycle:
        current = self.get(cycle_id)
        if current is None:
            raise KeyError(cycle_id)
        completed_at = datetime.now(timezone.utc)
        self.database.execute(
            "UPDATE learning_cycles SET status=?, completed_at=?, result_summary_json=?, error_type=NULL, error_message=NULL WHERE cycle_id=?",
            (status.value, _iso(completed_at), _json(summary or {}), cycle_id),
        )
        self.database.commit()
        updated = self.get(cycle_id)
        assert updated is not None
        self.upsert_learning_state(LearningState(
            scope_key=updated.scope_key,
            last_successful_cycle_id=updated.cycle_id,
            last_dataset_version_id=updated.dataset_version_id,
            last_dataset_end=_dataset_end(self.database.connection, updated.dataset_version_id),
            last_training_run_id=updated.training_run_id,
            last_candidate_id=candidate_id,
            active_model_version_id=self.get_production_model_id(updated.scope_key),
            active_policy_version_id=self.get_production_policy_id(updated.scope_key),
            last_failure=None,
            last_trigger=updated.trigger,
            updated_at=completed_at,
        ))
        return self.get(cycle_id)  # type: ignore[return-value]

    def get_production_model_id(self, scope_key: str) -> str | None:
        row = self.database.execute("SELECT model_version_id FROM learning_production_state WHERE scope_key=?", (scope_key,)).fetchone()
        return None if row is None else row["model_version_id"]

    def get_production_policy_id(self, scope_key: str) -> str | None:
        row = self.database.execute("SELECT policy_version_id FROM learning_production_state WHERE scope_key=?", (scope_key,)).fetchone()
        return None if row is None else row["policy_version_id"]

    def fail(self, cycle_id: str, *, error_type: str, error_message: str, error_stage: str) -> LearningCycle:
        self.database.execute(
            """UPDATE learning_cycles SET status = 'FAILED', completed_at = ?, error_type = ?, error_message = ?, result_summary_json = ? WHERE cycle_id = ?""",
            (_iso(datetime.now(timezone.utc)), error_type, error_message, _json({"error_stage": error_stage}), cycle_id),
        )
        self.database.commit()
        updated = self.get(cycle_id)
        if updated is not None:
            previous = self.get_learning_state(updated.scope_key)
            now = datetime.now(timezone.utc)
            self.upsert_learning_state(LearningState(
                scope_key=updated.scope_key,
                # A failed cycle must not erase the last known-good learning state.
                last_successful_cycle_id=None if previous is None else previous.last_successful_cycle_id,
                last_dataset_version_id=(
                    updated.dataset_version_id
                    if updated.dataset_version_id is not None
                    else (None if previous is None else previous.last_dataset_version_id)
                ),
                last_dataset_end=(
                    _dataset_end(self.database.connection, updated.dataset_version_id)
                    if updated.dataset_version_id is not None
                    else (None if previous is None else previous.last_dataset_end)
                ),
                last_training_run_id=(
                    updated.training_run_id
                    if updated.training_run_id is not None
                    else (None if previous is None else previous.last_training_run_id)
                ),
                last_candidate_id=(None if previous is None else previous.last_candidate_id),
                active_model_version_id=self.get_production_model_id(updated.scope_key),
                active_policy_version_id=self.get_production_policy_id(updated.scope_key),
                last_failure=f"{error_type}: {error_message}",
                last_trigger=updated.trigger,
                updated_at=now,
            ))
        return updated  # type: ignore[return-value]

    def _set_field(self, cycle_id: str, field_name: str, value: str | None) -> LearningCycle:
        allowed = {
            "dataset_version_id", "training_run_id", "model_version_id", "policy_version_id",
            "validation_evaluation_id", "oos_evaluation_id", "robustness_evaluation_id",
        }
        if field_name not in allowed:
            raise L10RepositoryError("unsupported cycle field")
        self.database.execute(f"UPDATE learning_cycles SET {field_name} = ? WHERE cycle_id = ?", (value, cycle_id))
        self.database.commit()
        return self.get(cycle_id)  # type: ignore[return-value]

    def set_dataset(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "dataset_version_id", value)
    def set_training_run(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "training_run_id", value)
    def set_model_version(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "model_version_id", value)
    def set_policy_version(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "policy_version_id", value)
    def set_validation_evaluation(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "validation_evaluation_id", value)
    def set_oos_evaluation(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "oos_evaluation_id", value)
    def set_robustness_evaluation(self, cycle_id: str, value: str) -> LearningCycle: return self._set_field(cycle_id, "robustness_evaluation_id", value)

    def create_candidate(
        self,
        *,
        cycle: LearningCycle,
        model_version_id: str | None,
        policy_version_id: str,
        dataset_version_id: str,
        validation_evaluation_id: str | None,
        oos_evaluation_id: str | None,
        robustness_evaluation_id: str | None,
        evidence: Mapping[str, Any] | None = None,
    ) -> LearningCandidate:
        payload = {
            "cycle_id": cycle.cycle_id,
            "model_version_id": model_version_id,
            "policy_version_id": policy_version_id,
            "dataset_version_id": dataset_version_id,
            "strategy_name": cycle.strategy_name,
            "strategy_variant": cycle.strategy_variant,
            "strategy_version": cycle.strategy_version,
            "validation_evaluation_id": validation_evaluation_id,
            "oos_evaluation_id": oos_evaluation_id,
            "robustness_evaluation_id": robustness_evaluation_id,
        }
        identity = _sha256(payload)
        candidate = LearningCandidate(
            candidate_id=f"cand-{identity[:32]}",
            identity_hash=identity,
            cycle_id=cycle.cycle_id,
            scope_key=cycle.scope_key,
            strategy_name=cycle.strategy_name,
            strategy_variant=cycle.strategy_variant,
            strategy_version=cycle.strategy_version,
            model_version_id=model_version_id,
            policy_version_id=policy_version_id,
            dataset_version_id=dataset_version_id,
            validation_evaluation_id=validation_evaluation_id,
            oos_evaluation_id=oos_evaluation_id,
            robustness_evaluation_id=robustness_evaluation_id,
            status=CandidateStatus.CANDIDATE,
            gate_eligible=False,
            gate_reasons=(),
            created_at=datetime.now(timezone.utc),
            evidence=dict(evidence or {}),
        )
        existing = self.get_candidate(candidate.candidate_id)
        if existing is not None:
            return existing
        self.database.execute(
            """INSERT INTO learning_candidates(
                candidate_id, identity_hash, cycle_id, scope_key, strategy_name, strategy_variant,
                strategy_version, model_version_id, policy_version_id, dataset_version_id,
                validation_evaluation_id, oos_evaluation_id, robustness_evaluation_id, status,
                gate_eligible, gate_reasons_json, created_at, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate.candidate_id, candidate.identity_hash, candidate.cycle_id, candidate.scope_key,
                candidate.strategy_name, candidate.strategy_variant, candidate.strategy_version,
                candidate.model_version_id, candidate.policy_version_id, candidate.dataset_version_id,
                candidate.validation_evaluation_id, candidate.oos_evaluation_id, candidate.robustness_evaluation_id,
                candidate.status.value, 0, _json([]), _iso(candidate.created_at), _json(candidate.evidence),
            ),
        )
        self.database.commit()
        return self.get_candidate(candidate.candidate_id)  # type: ignore[return-value]


    def list_candidates(self, *, scope_key: str | None = None, status: CandidateStatus | None = None) -> tuple[LearningCandidate, ...]:
        sql = "SELECT * FROM learning_candidates"
        params: list[str] = []
        conditions: list[str] = []
        if scope_key is not None:
            conditions.append("scope_key=?")
            params.append(scope_key)
        if status is not None:
            conditions.append("status=?")
            params.append(status.value)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at ASC, candidate_id ASC"
        rows = self.database.execute(sql, tuple(params)).fetchall()
        return tuple(_candidate_from_row(row) for row in rows)

    def list_audit_events(self, *, scope_key: str | None = None) -> tuple[Mapping[str, Any], ...]:
        if scope_key is None:
            rows = self.database.execute("SELECT * FROM learning_audit_events ORDER BY created_at ASC, event_id ASC").fetchall()
        else:
            rows = self.database.execute("SELECT * FROM learning_audit_events WHERE scope_key=? ORDER BY created_at ASC, event_id ASC", (scope_key,)).fetchall()
        return tuple({
            "event_id": row["event_id"],
            "cycle_id": row["cycle_id"],
            "scope_key": row["scope_key"],
            "action": row["action"],
            "actor_source": row["actor_source"],
            "model_version_id": row["model_version_id"],
            "policy_version_id": row["policy_version_id"],
            "dataset_version_id": row["dataset_version_id"],
            "reason": row["reason"],
            "status": row["status"],
            "details": json.loads(row["details_json"]),
            "created_at": row["created_at"],
        } for row in rows)

    def list_monitoring_snapshots(self, *, scope_key: str | None = None) -> tuple[MonitoringSnapshot, ...]:
        if scope_key is None:
            rows = self.database.execute("SELECT * FROM learning_monitoring_snapshots ORDER BY observed_at ASC, snapshot_id ASC").fetchall()
        else:
            rows = self.database.execute("SELECT * FROM learning_monitoring_snapshots WHERE scope_key=? ORDER BY observed_at ASC, snapshot_id ASC", (scope_key,)).fetchall()
        return tuple(_monitor_from_row(row) for row in rows)

    def list_production_history(self, *, scope_key: str | None = None) -> tuple[PromotionHistoryEntry, ...]:
        if scope_key is None:
            rows = self.database.execute("SELECT * FROM learning_production_history ORDER BY created_at ASC, history_id ASC").fetchall()
        else:
            rows = self.database.execute("SELECT * FROM learning_production_history WHERE scope_key=? ORDER BY created_at ASC, history_id ASC", (scope_key,)).fetchall()
        return tuple(PromotionHistoryEntry(
            history_id=row["history_id"], event_type=row["event_type"], scope_key=row["scope_key"], cycle_id=row["cycle_id"],
            old_model_version_id=row["old_model_version_id"], new_model_version_id=row["new_model_version_id"],
            old_policy_version_id=row["old_policy_version_id"], new_policy_version_id=row["new_policy_version_id"],
            reason=row["reason"], policy_version=row["policy_version"], evidence=json.loads(row["evidence_json"]),
            status=row["status"], created_at=_dt(row["created_at"]),
        ) for row in rows)

    def get_candidate(self, candidate_id: str) -> LearningCandidate | None:
        row = self.database.execute("SELECT * FROM learning_candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        return None if row is None else _candidate_from_row(row)

    def record_promotion_gate(self, candidate_id: str, eligible: bool, reasons: tuple[str, ...]) -> LearningCandidate:
        self.database.execute(
            "UPDATE learning_candidates SET gate_eligible = ?, gate_reasons_json = ? WHERE candidate_id = ?",
            (1 if eligible else 0, _json(list(reasons)), candidate_id),
        )
        self.database.commit()
        return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def get_candidate_evidence(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.database.execute("SELECT evidence_json FROM learning_candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        return None if row is None else json.loads(row["evidence_json"])

    def mark_challenger(self, candidate_id: str) -> LearningCandidate:
        self.database.execute("UPDATE learning_candidates SET status = 'CHALLENGER' WHERE candidate_id = ?", (candidate_id,))
        self.database.commit()
        return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def reject_candidate(self, candidate_id: str, *, reason: str) -> LearningCandidate:
        self.database.execute("UPDATE learning_candidates SET status = 'REJECTED', gate_reasons_json = ? WHERE candidate_id = ?", (_json([reason]), candidate_id))
        self.database.commit()
        return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def mark_candidate_status(self, candidate_id: str, status: CandidateStatus) -> LearningCandidate:
        self.database.execute("UPDATE learning_candidates SET status = ? WHERE candidate_id = ?", (status.value, candidate_id))
        self.database.commit()
        return self.get_candidate(candidate_id)  # type: ignore[return-value]

    def audit(self, action: str, cycle: LearningCycle | None, *, reason: str, status: str) -> None:
        scope_key = None if cycle is None else cycle.scope_key
        details = {"reason": reason, "status": status}
        self.database.execute(
            "INSERT INTO learning_audit_events(event_id, cycle_id, scope_key, action, actor_source, model_version_id, policy_version_id, dataset_version_id, reason, status, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"aud-{_sha256({'cycle': None if cycle is None else cycle.cycle_id, 'action': action, 'reason': reason, 'status': status, 'at': datetime.now(timezone.utc).isoformat()})[:32]}",
                None if cycle is None else cycle.cycle_id,
                scope_key,
                action,
                "SYSTEM",
                None if cycle is None else cycle.model_version_id,
                None if cycle is None else cycle.policy_version_id,
                None if cycle is None else cycle.dataset_version_id,
                reason,
                status,
                _json(details),
                _iso(datetime.now(timezone.utc)),
            ),
        )
        self.database.commit()

    def get_l8(self, evaluation_id: str | None) -> Any | None:
        if evaluation_id is None:
            return None
        try:
            from app.learning.advanced_evaluation_repository import L8EvaluationRepository
            return L8EvaluationRepository(self.database).get(evaluation_id)
        except Exception:
            return None

    def count_completed_since(self, scope_key: str, since: datetime | None) -> int:
        strategy_name, strategy_variant, strategy_version = scope_key.split("::", 2)
        actual_name = {"CLASSIC": "Classic", "SMC": "SMC", "ICT": "ICT", "AI": "AI"}.get(strategy_name, strategy_name)
        if since is None:
            row = self.database.execute(
                "SELECT COUNT(*) AS c FROM learning_records WHERE status = 'COMPLETED' AND strategy_name = ? AND strategy_variant = ? AND strategy_version = ?",
                (actual_name, strategy_variant, strategy_version),
            ).fetchone()
        else:
            row = self.database.execute(
                "SELECT COUNT(*) AS c FROM learning_records WHERE status = 'COMPLETED' AND strategy_name = ? AND strategy_variant = ? AND strategy_version = ? AND decision_timestamp > ?",
                (actual_name, strategy_variant, strategy_version, _iso(since)),
            ).fetchone()
        return int(row["c"])

    def get_learning_state(self, scope_key: str) -> LearningState | None:
        row = self.database.execute("SELECT * FROM learning_state WHERE scope_key = ?", (scope_key,)).fetchone()
        return None if row is None else _state_from_row(row)

    def upsert_learning_state(self, state: LearningState) -> None:
        self.database.execute(
            """INSERT INTO learning_state(scope_key, last_successful_cycle_id, last_dataset_version_id, last_dataset_end, last_training_run_id, last_candidate_id, active_model_version_id, active_policy_version_id, last_failure, last_trigger, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
                last_successful_cycle_id=excluded.last_successful_cycle_id,
                last_dataset_version_id=excluded.last_dataset_version_id,
                last_dataset_end=excluded.last_dataset_end,
                last_training_run_id=excluded.last_training_run_id,
                last_candidate_id=excluded.last_candidate_id,
                active_model_version_id=excluded.active_model_version_id,
                active_policy_version_id=excluded.active_policy_version_id,
                last_failure=excluded.last_failure,
                last_trigger=excluded.last_trigger,
                updated_at=excluded.updated_at""",
            (
                state.scope_key, state.last_successful_cycle_id, state.last_dataset_version_id,
                _iso(state.last_dataset_end), state.last_training_run_id, state.last_candidate_id,
                state.active_model_version_id, state.active_policy_version_id, state.last_failure,
                state.last_trigger, _iso(state.updated_at),
            ),
        )
        self.database.commit()

    def create_monitoring_snapshot(self, snapshot: MonitoringSnapshot) -> MonitoringSnapshot:
        existing = self.database.execute("SELECT * FROM learning_monitoring_snapshots WHERE snapshot_id = ?", (snapshot.snapshot_id,)).fetchone()
        if existing is not None:
            return _monitor_from_row(existing)
        self.database.execute(
            "INSERT INTO learning_monitoring_snapshots(snapshot_id, scope_key, model_version_id, policy_version_id, observed_at, sample_count, metrics_json, baseline_metrics_json, drift_score, drift_detected, runtime_error_count, artifact_healthy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_id, snapshot.scope_key, snapshot.model_version_id, snapshot.policy_version_id,
                _iso(snapshot.observed_at), snapshot.sample_count, _json(snapshot.metrics), _json(snapshot.baseline_metrics),
                snapshot.drift_score, 1 if snapshot.drift_detected else 0, snapshot.runtime_error_count, 1 if snapshot.artifact_healthy else 0,
            ),
        )
        self.database.commit()
        return snapshot


class ProductionStateRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get(self, scope_key: str) -> ProductionModelState | None:
        row = self.database.execute("SELECT * FROM learning_production_state WHERE scope_key = ?", (scope_key,)).fetchone()
        return None if row is None else _production_from_row(row)

    def get_learning_state(self, scope_key: str) -> LearningState | None:
        return LearningCycleRepository(self.database).get_learning_state(scope_key)

    def initialize(
        self,
        *,
        scope_key: str,
        strategy_name: str,
        strategy_variant: str,
        strategy_version: str,
        model_version_id: str | None,
        policy_version_id: str,
        reason: str,
        now: datetime,
    ) -> ProductionModelState:
        existing = self.get(scope_key)
        if existing is not None and existing.status == "ACTIVE":
            raise L10RepositoryError("production state already initialized")
        state = ProductionModelState(
            scope_key=scope_key,
            strategy_name=strategy_name,
            strategy_variant=strategy_variant,
            strategy_version=strategy_version,
            model_version_id=model_version_id,
            policy_version_id=policy_version_id,
            activated_at=now,
            activation_reason=reason,
            source_cycle_id=None,
            previous_model_version_id=None,
            previous_policy_version_id=None,
            status="ACTIVE",
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO learning_production_state(scope_key, strategy_name, strategy_variant, strategy_version, model_version_id, policy_version_id, activated_at, activation_reason, source_cycle_id, previous_model_version_id, previous_policy_version_id, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (state.scope_key, state.strategy_name, state.strategy_variant, state.strategy_version, state.model_version_id, state.policy_version_id, _iso(state.activated_at), state.activation_reason, state.source_cycle_id, state.previous_model_version_id, state.previous_policy_version_id, state.status),
            )
            conn.execute(
                "INSERT INTO learning_production_history(history_id, event_type, scope_key, cycle_id, old_model_version_id, new_model_version_id, old_policy_version_id, new_policy_version_id, reason, policy_version, evidence_json, status, created_at) VALUES (?, 'INITIAL_ACTIVATION', ?, NULL, NULL, ?, NULL, ?, ?, NULL, '{}', 'COMPLETED', ?)",
                (f"hist-{_sha256(state.__dict__)[:32]}", state.scope_key, state.model_version_id, state.policy_version_id, reason, _iso(now)),
            )
        return state

    def promote(
        self,
        *,
        candidate: LearningCandidate,
        cycle: LearningCycle,
        new_model_version_id: str | None,
        new_policy_version_id: str,
        reason: str,
        promotion_policy_version: str,
        evidence: Mapping[str, Any],
    ) -> ProductionModelState:
        current = self.get(cycle.scope_key)
        if current is None or current.status != "ACTIVE":
            raise L10RepositoryError("active Champion is required for atomic promotion")
        now = datetime.now(timezone.utc)
        new_state = ProductionModelState(
            scope_key=cycle.scope_key,
            strategy_name=cycle.strategy_name,
            strategy_variant=cycle.strategy_variant,
            strategy_version=cycle.strategy_version,
            model_version_id=new_model_version_id,
            policy_version_id=new_policy_version_id,
            activated_at=now,
            activation_reason=reason,
            source_cycle_id=cycle.cycle_id,
            previous_model_version_id=current.model_version_id,
            previous_policy_version_id=current.policy_version_id,
            status="ACTIVE",
        )
        history_id = f"hist-{_sha256({'cycle': cycle.cycle_id, 'old': current.model_version_id, 'new': new_model_version_id, 'policy': new_policy_version_id})[:32]}"
        with self.database.transaction() as conn:
            conn.execute(
                "UPDATE learning_production_state SET strategy_name=?, strategy_variant=?, strategy_version=?, model_version_id=?, policy_version_id=?, activated_at=?, activation_reason=?, source_cycle_id=?, previous_model_version_id=?, previous_policy_version_id=?, status='ACTIVE' WHERE scope_key=?",
                (new_state.strategy_name, new_state.strategy_variant, new_state.strategy_version, new_state.model_version_id, new_state.policy_version_id, _iso(now), reason, cycle.cycle_id, current.model_version_id, current.policy_version_id, cycle.scope_key),
            )
            conn.execute(
                "INSERT INTO learning_production_history(history_id, event_type, scope_key, cycle_id, old_model_version_id, new_model_version_id, old_policy_version_id, new_policy_version_id, reason, policy_version, evidence_json, status, created_at) VALUES (?, 'PROMOTION', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', ?)",
                (history_id, cycle.scope_key, cycle.cycle_id, current.model_version_id, new_model_version_id, current.policy_version_id, new_policy_version_id, reason, promotion_policy_version, _json(evidence), _iso(now)),
            )
            conn.execute("UPDATE learning_candidates SET status='PROMOTED' WHERE candidate_id=?", (candidate.candidate_id,))
            _upsert_state_conn(conn, LearningState(
                scope_key=cycle.scope_key,
                last_successful_cycle_id=cycle.cycle_id,
                last_dataset_version_id=cycle.dataset_version_id,
                last_dataset_end=_dataset_end(conn, cycle.dataset_version_id),
                last_training_run_id=cycle.training_run_id,
                last_candidate_id=candidate.candidate_id,
                active_model_version_id=new_model_version_id,
                active_policy_version_id=new_policy_version_id,
                last_failure=None,
                last_trigger=cycle.trigger,
                updated_at=now,
            ))
        return new_state

    def rollback(self, scope_key: str, *, reason: str, evidence: Mapping[str, Any]) -> ProductionModelState:
        current = self.get(scope_key)
        if current is None or current.previous_model_version_id is None and current.previous_policy_version_id is None:
            raise L10RepositoryError("no previous Champion is available for rollback")
        now = datetime.now(timezone.utc)
        restored = ProductionModelState(
            scope_key=current.scope_key,
            strategy_name=current.strategy_name,
            strategy_variant=current.strategy_variant,
            strategy_version=current.strategy_version,
            model_version_id=current.previous_model_version_id,
            policy_version_id=current.previous_policy_version_id or "",
            activated_at=now,
            activation_reason=reason,
            source_cycle_id=current.source_cycle_id,
            previous_model_version_id=current.model_version_id,
            previous_policy_version_id=current.policy_version_id,
            status="ACTIVE",
        )
        history_id = f"hist-{_sha256({'scope': scope_key, 'rollback_to': restored.model_version_id, 'policy': restored.policy_version_id, 'reason': reason})[:32]}"
        with self.database.transaction() as conn:
            conn.execute(
                "UPDATE learning_production_state SET model_version_id=?, policy_version_id=?, activated_at=?, activation_reason=?, previous_model_version_id=?, previous_policy_version_id=?, status='ACTIVE' WHERE scope_key=?",
                (restored.model_version_id, restored.policy_version_id, _iso(now), reason, current.model_version_id, current.policy_version_id, scope_key),
            )
            conn.execute(
                "INSERT INTO learning_production_history(history_id, event_type, scope_key, cycle_id, old_model_version_id, new_model_version_id, old_policy_version_id, new_policy_version_id, reason, policy_version, evidence_json, status, created_at) VALUES (?, 'ROLLBACK', ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'COMPLETED', ?)",
                (history_id, scope_key, current.source_cycle_id, current.model_version_id, restored.model_version_id, current.policy_version_id, restored.policy_version_id, reason, _json(evidence), _iso(now)),
            )
        return restored


# ---------------- row helpers ----------------
def _cycle_from_row(row: Any) -> LearningCycle:
    return LearningCycle(
        cycle_id=row["cycle_id"], identity_hash=row["identity_hash"], scope_key=row["scope_key"],
        strategy_name=row["strategy_name"], strategy_variant=row["strategy_variant"], strategy_version=row["strategy_version"],
        trigger=row["trigger_type"], trigger_key=row["trigger_key"], status=LearningCycleStatus(row["status"]),
        started_at=_dt(row["started_at"]), completed_at=None if row["completed_at"] is None else _dt(row["completed_at"]),
        dataset_version_id=row["dataset_version_id"], training_run_id=row["training_run_id"], model_version_id=row["model_version_id"],
        policy_version_id=row["policy_version_id"], validation_evaluation_id=row["validation_evaluation_id"],
        oos_evaluation_id=row["oos_evaluation_id"], robustness_evaluation_id=row["robustness_evaluation_id"],
        configuration_fingerprint=row["configuration_fingerprint"], retry_count=int(row["retry_count"]),
        result_summary=json.loads(row["result_summary_json"]), error_type=row["error_type"], error_message=row["error_message"],
    )


def _candidate_from_row(row: Any) -> LearningCandidate:
    return LearningCandidate(
        candidate_id=row["candidate_id"], identity_hash=row["identity_hash"], cycle_id=row["cycle_id"], scope_key=row["scope_key"],
        strategy_name=row["strategy_name"], strategy_variant=row["strategy_variant"], strategy_version=row["strategy_version"],
        model_version_id=row["model_version_id"], policy_version_id=row["policy_version_id"], dataset_version_id=row["dataset_version_id"],
        validation_evaluation_id=row["validation_evaluation_id"], oos_evaluation_id=row["oos_evaluation_id"], robustness_evaluation_id=row["robustness_evaluation_id"],
        status=CandidateStatus(row["status"]), gate_eligible=bool(row["gate_eligible"]), gate_reasons=tuple(json.loads(row["gate_reasons_json"])),
        created_at=_dt(row["created_at"]), evidence=json.loads(row["evidence_json"]),
    )


def _production_from_row(row: Any) -> ProductionModelState:
    return ProductionModelState(
        scope_key=row["scope_key"], strategy_name=row["strategy_name"], strategy_variant=row["strategy_variant"], strategy_version=row["strategy_version"],
        model_version_id=row["model_version_id"], policy_version_id=row["policy_version_id"], activated_at=_dt(row["activated_at"]),
        activation_reason=row["activation_reason"], source_cycle_id=row["source_cycle_id"], previous_model_version_id=row["previous_model_version_id"],
        previous_policy_version_id=row["previous_policy_version_id"], status=row["status"],
    )


def _monitor_from_row(row: Any) -> MonitoringSnapshot:
    return MonitoringSnapshot(
        snapshot_id=row["snapshot_id"], scope_key=row["scope_key"], model_version_id=row["model_version_id"], policy_version_id=row["policy_version_id"],
        observed_at=_dt(row["observed_at"]), sample_count=int(row["sample_count"]), metrics=json.loads(row["metrics_json"]), baseline_metrics=json.loads(row["baseline_metrics_json"]),
        drift_score=float(row["drift_score"]), drift_detected=bool(row["drift_detected"]), runtime_error_count=int(row["runtime_error_count"]), artifact_healthy=bool(row["artifact_healthy"]),
    )


def _state_from_row(row: Any) -> LearningState:
    return LearningState(
        scope_key=row["scope_key"], last_successful_cycle_id=row["last_successful_cycle_id"], last_dataset_version_id=row["last_dataset_version_id"],
        last_dataset_end=None if row["last_dataset_end"] is None else _dt(row["last_dataset_end"]), last_training_run_id=row["last_training_run_id"],
        last_candidate_id=row["last_candidate_id"], active_model_version_id=row["active_model_version_id"], active_policy_version_id=row["active_policy_version_id"],
        last_failure=row["last_failure"], last_trigger=row["last_trigger"], updated_at=_dt(row["updated_at"]),
    )


def _upsert_state_conn(conn: Any, state: LearningState) -> None:
    conn.execute(
        """INSERT INTO learning_state(scope_key, last_successful_cycle_id, last_dataset_version_id, last_dataset_end, last_training_run_id, last_candidate_id, active_model_version_id, active_policy_version_id, last_failure, last_trigger, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope_key) DO UPDATE SET last_successful_cycle_id=excluded.last_successful_cycle_id, last_dataset_version_id=excluded.last_dataset_version_id, last_dataset_end=excluded.last_dataset_end, last_training_run_id=excluded.last_training_run_id, last_candidate_id=excluded.last_candidate_id, active_model_version_id=excluded.active_model_version_id, active_policy_version_id=excluded.active_policy_version_id, last_failure=excluded.last_failure, last_trigger=excluded.last_trigger, updated_at=excluded.updated_at""",
        (state.scope_key, state.last_successful_cycle_id, state.last_dataset_version_id, _iso(state.last_dataset_end), state.last_training_run_id, state.last_candidate_id, state.active_model_version_id, state.active_policy_version_id, state.last_failure, state.last_trigger, _iso(state.updated_at)),
    )


def _dataset_end(conn: Any, dataset_version_id: str | None) -> datetime | None:
    if dataset_version_id is None:
        return None
    row = conn.execute("SELECT end_timestamp FROM learning_dataset_versions WHERE dataset_version_id = ?", (dataset_version_id,)).fetchone()
    return None if row is None or row["end_timestamp"] is None else _dt(row["end_timestamp"])


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _iso(value)
    return value


def _json(value: Any) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "CandidateStatus", "L10RepositoryError", "LearningCandidate", "LearningCycle", "LearningCycleRepository",
    "LearningCycleStatus", "LearningState", "MonitoringSnapshot", "ProductionModelState", "ProductionStateRepository",
    "PromotionHistoryEntry",
]

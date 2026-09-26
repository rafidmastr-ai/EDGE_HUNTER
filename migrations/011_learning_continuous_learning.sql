-- EDGE HUNTER L10: controlled continuous-learning, candidate and production state.
CREATE TABLE IF NOT EXISTS learning_cycles (
    cycle_id TEXT PRIMARY KEY,
    identity_hash TEXT NOT NULL UNIQUE,
    scope_key TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('SCHEDULED','DATA_THRESHOLD','MANUAL','DRIFT')),
    trigger_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'DATA_COLLECTION','DATA_COMPLETION','DATASET_BUILD','TRAINING','REGISTRATION',
        'VALIDATION','OOS','ROBUSTNESS','CANDIDATE','PROMOTION_GATE','PROMOTED',
        'MONITORING','ROLLED_BACK','FAILED'
    )),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    dataset_version_id TEXT REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    training_run_id TEXT REFERENCES learning_training_runs(training_run_id) ON DELETE RESTRICT,
    model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    policy_version_id TEXT REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    validation_evaluation_id TEXT REFERENCES learning_model_evaluations(evaluation_id) ON DELETE RESTRICT,
    oos_evaluation_id TEXT REFERENCES learning_l8_evaluations(evaluation_id) ON DELETE RESTRICT,
    robustness_evaluation_id TEXT REFERENCES learning_l8_evaluations(evaluation_id) ON DELETE RESTRICT,
    configuration_fingerprint TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    result_summary_json TEXT NOT NULL DEFAULT '{}',
    error_type TEXT,
    error_message TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_cycles_active_scope
    ON learning_cycles(scope_key)
    WHERE status IN ('DATA_COLLECTION','DATA_COMPLETION','DATASET_BUILD','TRAINING','REGISTRATION','VALIDATION','OOS','ROBUSTNESS','PROMOTION_GATE','MONITORING');
CREATE INDEX IF NOT EXISTS idx_learning_cycles_scope ON learning_cycles(scope_key, started_at, cycle_id);
CREATE INDEX IF NOT EXISTS idx_learning_cycles_trigger ON learning_cycles(trigger_type, trigger_key);

CREATE TABLE IF NOT EXISTS learning_candidates (
    candidate_id TEXT PRIMARY KEY,
    identity_hash TEXT NOT NULL UNIQUE,
    cycle_id TEXT NOT NULL REFERENCES learning_cycles(cycle_id) ON DELETE RESTRICT,
    scope_key TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    policy_version_id TEXT NOT NULL REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    validation_evaluation_id TEXT REFERENCES learning_model_evaluations(evaluation_id) ON DELETE RESTRICT,
    oos_evaluation_id TEXT REFERENCES learning_l8_evaluations(evaluation_id) ON DELETE RESTRICT,
    robustness_evaluation_id TEXT REFERENCES learning_l8_evaluations(evaluation_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('CANDIDATE','CHALLENGER','PROMOTED','REJECTED','ROLLED_BACK')),
    gate_eligible INTEGER NOT NULL DEFAULT 0 CHECK (gate_eligible IN (0,1)),
    gate_reasons_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_learning_candidates_scope ON learning_candidates(scope_key, created_at, candidate_id);
CREATE INDEX IF NOT EXISTS idx_learning_candidates_status ON learning_candidates(status, created_at);

CREATE TABLE IF NOT EXISTS learning_production_state (
    scope_key TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    policy_version_id TEXT REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    activated_at TEXT NOT NULL,
    activation_reason TEXT NOT NULL,
    source_cycle_id TEXT REFERENCES learning_cycles(cycle_id) ON DELETE RESTRICT,
    previous_model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    previous_policy_version_id TEXT REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('BASE_ONLY','ACTIVE'))
);

CREATE TABLE IF NOT EXISTS learning_production_history (
    history_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (event_type IN ('INITIAL_ACTIVATION','PROMOTION','ROLLBACK')),
    scope_key TEXT NOT NULL,
    cycle_id TEXT REFERENCES learning_cycles(cycle_id) ON DELETE RESTRICT,
    old_model_version_id TEXT,
    new_model_version_id TEXT,
    old_policy_version_id TEXT,
    new_policy_version_id TEXT,
    reason TEXT NOT NULL,
    policy_version TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('ATTEMPTED','COMPLETED','FAILED')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_production_history_scope ON learning_production_history(scope_key, created_at, history_id);

CREATE TABLE IF NOT EXISTS learning_monitoring_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    policy_version_id TEXT REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    observed_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
    metrics_json TEXT NOT NULL,
    baseline_metrics_json TEXT NOT NULL,
    drift_score REAL NOT NULL CHECK (drift_score >= 0),
    drift_detected INTEGER NOT NULL CHECK (drift_detected IN (0,1)),
    runtime_error_count INTEGER NOT NULL CHECK (runtime_error_count >= 0),
    artifact_healthy INTEGER NOT NULL CHECK (artifact_healthy IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_learning_monitoring_scope ON learning_monitoring_snapshots(scope_key, observed_at, snapshot_id);

CREATE TABLE IF NOT EXISTS learning_state (
    scope_key TEXT PRIMARY KEY,
    last_successful_cycle_id TEXT REFERENCES learning_cycles(cycle_id) ON DELETE RESTRICT,
    last_dataset_version_id TEXT REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    last_dataset_end TEXT,
    last_training_run_id TEXT REFERENCES learning_training_runs(training_run_id) ON DELETE RESTRICT,
    last_candidate_id TEXT REFERENCES learning_candidates(candidate_id) ON DELETE RESTRICT,
    active_model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    active_policy_version_id TEXT REFERENCES learning_policy_versions(policy_version_id) ON DELETE RESTRICT,
    last_failure TEXT,
    last_trigger TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_audit_events (
    event_id TEXT PRIMARY KEY,
    cycle_id TEXT REFERENCES learning_cycles(cycle_id) ON DELETE RESTRICT,
    scope_key TEXT,
    action TEXT NOT NULL,
    actor_source TEXT NOT NULL,
    model_version_id TEXT,
    policy_version_id TEXT,
    dataset_version_id TEXT,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_audit_cycle ON learning_audit_events(cycle_id, created_at, event_id);
CREATE INDEX IF NOT EXISTS idx_learning_audit_scope ON learning_audit_events(scope_key, created_at, event_id);

CREATE TRIGGER IF NOT EXISTS trg_learning_cycle_identity_immutable
BEFORE UPDATE ON learning_cycles
BEGIN
    SELECT CASE WHEN
        NEW.cycle_id <> OLD.cycle_id OR NEW.identity_hash <> OLD.identity_hash OR NEW.scope_key <> OLD.scope_key OR
        NEW.strategy_name <> OLD.strategy_name OR NEW.strategy_variant <> OLD.strategy_variant OR NEW.strategy_version <> OLD.strategy_version OR
        NEW.trigger_type <> OLD.trigger_type OR NEW.trigger_key <> OLD.trigger_key OR NEW.started_at <> OLD.started_at OR
        NEW.configuration_fingerprint <> OLD.configuration_fingerprint
    THEN RAISE(ABORT, 'learning cycle identity is immutable') END;
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_candidate_identity_immutable
BEFORE UPDATE ON learning_candidates
BEGIN
    SELECT CASE WHEN
        NEW.candidate_id <> OLD.candidate_id OR NEW.identity_hash <> OLD.identity_hash OR NEW.cycle_id <> OLD.cycle_id OR
        NEW.scope_key <> OLD.scope_key OR NEW.strategy_name <> OLD.strategy_name OR NEW.strategy_variant <> OLD.strategy_variant OR
        NEW.strategy_version <> OLD.strategy_version OR COALESCE(NEW.model_version_id,'') <> COALESCE(OLD.model_version_id,'') OR
        NEW.policy_version_id <> OLD.policy_version_id OR NEW.dataset_version_id <> OLD.dataset_version_id OR
        COALESCE(NEW.validation_evaluation_id,'') <> COALESCE(OLD.validation_evaluation_id,'') OR
        COALESCE(NEW.oos_evaluation_id,'') <> COALESCE(OLD.oos_evaluation_id,'') OR
        COALESCE(NEW.robustness_evaluation_id,'') <> COALESCE(OLD.robustness_evaluation_id,'') OR
        NEW.created_at <> OLD.created_at
    THEN RAISE(ABORT, 'learning candidate identity is immutable') END;
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_production_history_no_update
BEFORE UPDATE ON learning_production_history
BEGIN
    SELECT RAISE(ABORT, 'production history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_learning_production_history_no_delete
BEFORE DELETE ON learning_production_history
BEGIN
    SELECT RAISE(ABORT, 'production history is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_monitoring_no_update
BEFORE UPDATE ON learning_monitoring_snapshots
BEGIN
    SELECT RAISE(ABORT, 'monitoring snapshots are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_learning_monitoring_no_delete
BEFORE DELETE ON learning_monitoring_snapshots
BEGIN
    SELECT RAISE(ABORT, 'monitoring snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_audit_no_update
BEFORE UPDATE ON learning_audit_events
BEGIN
    SELECT RAISE(ABORT, 'learning audit events are append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_learning_audit_no_delete
BEFORE DELETE ON learning_audit_events
BEGIN
    SELECT RAISE(ABORT, 'learning audit events are append-only');
END;


CREATE TRIGGER IF NOT EXISTS trg_learning_cycle_no_delete
BEFORE DELETE ON learning_cycles
BEGIN
    SELECT RAISE(ABORT, 'learning cycle history is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_candidate_no_delete
BEFORE DELETE ON learning_candidates
BEGIN
    SELECT RAISE(ABORT, 'learning candidate history is immutable');
END;

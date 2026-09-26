CREATE TABLE IF NOT EXISTS learning_model_versions (
    model_version_id TEXT PRIMARY KEY,
    model_version_label TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    training_run_id TEXT NOT NULL REFERENCES learning_training_runs(training_run_id) ON DELETE RESTRICT,
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    feature_schema_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    artifact_format_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('REGISTERED', 'EVALUATED', 'REVOKED')),
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_model_versions_strategy
    ON learning_model_versions(strategy_name, strategy_version, registered_at, model_version_id);
CREATE INDEX IF NOT EXISTS idx_learning_model_versions_training_run
    ON learning_model_versions(training_run_id);
CREATE INDEX IF NOT EXISTS idx_learning_model_versions_dataset
    ON learning_model_versions(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_learning_model_versions_status
    ON learning_model_versions(status, registered_at);

CREATE TRIGGER IF NOT EXISTS trg_learning_model_versions_immutable_update
BEFORE UPDATE ON learning_model_versions
BEGIN
    SELECT CASE WHEN
        NEW.model_version_id <> OLD.model_version_id OR
        NEW.model_version_label <> OLD.model_version_label OR
        NEW.strategy_name <> OLD.strategy_name OR
        NEW.strategy_variant <> OLD.strategy_variant OR
        NEW.strategy_version <> OLD.strategy_version OR
        NEW.model_type <> OLD.model_type OR
        NEW.training_run_id <> OLD.training_run_id OR
        NEW.dataset_version_id <> OLD.dataset_version_id OR
        NEW.feature_schema_version <> OLD.feature_schema_version OR
        NEW.label_version <> OLD.label_version OR
        NEW.artifact_path <> OLD.artifact_path OR
        NEW.artifact_sha256 <> OLD.artifact_sha256 OR
        NEW.artifact_format_version <> OLD.artifact_format_version OR
        NEW.created_at <> OLD.created_at OR
        NEW.registered_at <> OLD.registered_at OR
        NEW.metadata_json <> OLD.metadata_json OR
        NEW.status NOT IN ('REGISTERED', 'EVALUATED', 'REVOKED')
    THEN RAISE(ABORT, 'ModelVersion immutable fields cannot be changed') END;
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_model_versions_immutable_delete
BEFORE DELETE ON learning_model_versions
BEGIN
    SELECT RAISE(ABORT, 'ModelVersion records are immutable');
END;

CREATE TABLE IF NOT EXISTS learning_model_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    model_version_id TEXT NOT NULL REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    split TEXT NOT NULL CHECK (split = 'VALIDATION'),
    evaluated_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    metrics_json TEXT NOT NULL,
    confusion_matrix_json TEXT NOT NULL,
    class_distribution_json TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    evaluation_config_json TEXT NOT NULL,
    evaluation_fingerprint TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error_type TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_learning_model_evaluations_model
    ON learning_model_evaluations(model_version_id, evaluated_at, evaluation_id);
CREATE INDEX IF NOT EXISTS idx_learning_model_evaluations_dataset
    ON learning_model_evaluations(dataset_version_id, evaluated_at, evaluation_id);

CREATE TRIGGER IF NOT EXISTS trg_learning_model_evaluations_immutable_update
BEFORE UPDATE ON learning_model_evaluations
BEGIN
    SELECT CASE WHEN
        NEW.evaluation_id <> OLD.evaluation_id OR
        NEW.model_version_id <> OLD.model_version_id OR
        NEW.dataset_version_id <> OLD.dataset_version_id OR
        NEW.split <> OLD.split OR
        NEW.feature_schema_version <> OLD.feature_schema_version OR
        NEW.label_version <> OLD.label_version OR
        NEW.evaluation_fingerprint <> OLD.evaluation_fingerprint OR
        NEW.started_at <> OLD.started_at OR
        NEW.status NOT IN ('RUNNING', 'COMPLETED', 'FAILED')
    THEN RAISE(ABORT, 'evaluation identity fields cannot be changed') END;
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_model_evaluations_immutable_delete
BEFORE DELETE ON learning_model_evaluations
BEGIN
    SELECT RAISE(ABORT, 'ModelEvaluation records are immutable');
END;

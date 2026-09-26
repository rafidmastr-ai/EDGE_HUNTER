CREATE TABLE IF NOT EXISTS learning_l8_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    evaluation_type TEXT NOT NULL CHECK (evaluation_type IN ('OOS', 'WALK_FORWARD', 'ROBUSTNESS', 'SENSITIVITY')),
    model_version_id TEXT NOT NULL REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    split TEXT NOT NULL CHECK (split = 'OOS'),
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('COMPLETED', 'FAILED', 'INSUFFICIENT_DATA', 'NOT_APPLICABLE')),
    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
    eligible_samples INTEGER NOT NULL CHECK (eligible_samples >= 0),
    excluded_samples INTEGER NOT NULL CHECK (excluded_samples >= 0),
    metrics_json TEXT NOT NULL,
    confusion_matrix_json TEXT NOT NULL,
    class_distribution_json TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    evaluation_config_json TEXT NOT NULL,
    evaluation_fingerprint TEXT NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL,
    windows_json TEXT NOT NULL,
    stability_summary_json TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_type TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    CHECK (eligible_samples <= sample_count),
    CHECK ((status = 'FAILED' AND error_message IS NOT NULL) OR status <> 'FAILED')
);

CREATE INDEX IF NOT EXISTS idx_learning_l8_evaluations_model
    ON learning_l8_evaluations(model_version_id, evaluated_at, evaluation_id);
CREATE INDEX IF NOT EXISTS idx_learning_l8_evaluations_dataset
    ON learning_l8_evaluations(dataset_version_id, evaluated_at, evaluation_id);
CREATE INDEX IF NOT EXISTS idx_learning_l8_evaluations_type
    ON learning_l8_evaluations(evaluation_type, evaluated_at, evaluation_id);
CREATE INDEX IF NOT EXISTS idx_learning_l8_evaluations_status
    ON learning_l8_evaluations(status, evaluated_at, evaluation_id);

CREATE TRIGGER IF NOT EXISTS trg_learning_l8_evaluations_immutable_update
BEFORE UPDATE ON learning_l8_evaluations
BEGIN
    SELECT CASE WHEN
        NEW.evaluation_id <> OLD.evaluation_id OR
        NEW.evaluation_type <> OLD.evaluation_type OR
        NEW.model_version_id <> OLD.model_version_id OR
        NEW.dataset_version_id <> OLD.dataset_version_id OR
        NEW.split <> OLD.split OR
        NEW.evaluation_fingerprint <> OLD.evaluation_fingerprint OR
        NEW.feature_schema_version <> OLD.feature_schema_version OR
        NEW.label_version <> OLD.label_version OR
        NEW.created_at <> OLD.created_at
    THEN RAISE(ABORT, 'L8 evaluation identity fields are immutable') END;
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_l8_evaluations_immutable_delete
BEFORE DELETE ON learning_l8_evaluations
BEGIN
    SELECT RAISE(ABORT, 'L8 evaluation records are immutable');
END;

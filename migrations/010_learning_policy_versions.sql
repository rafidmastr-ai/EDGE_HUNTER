CREATE TABLE IF NOT EXISTS learning_policy_versions (
    policy_version_id TEXT PRIMARY KEY,
    policy_version_label TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    policy_types_json TEXT NOT NULL,
    source_model_version_id TEXT REFERENCES learning_model_versions(model_version_id) ON DELETE RESTRICT,
    source_training_run_id TEXT REFERENCES learning_training_runs(training_run_id) ON DELETE RESTRICT,
    source_dataset_version_id TEXT REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    feature_schema_version TEXT,
    label_version TEXT,
    policy_parameters_json TEXT NOT NULL,
    enabled_capabilities_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    default_mode TEXT NOT NULL CHECK (default_mode IN ('DISABLED', 'SHADOW', 'CANDIDATE'))
);

CREATE INDEX IF NOT EXISTS idx_learning_policy_versions_strategy
    ON learning_policy_versions(strategy_name, strategy_version, created_at, policy_version_id);
CREATE INDEX IF NOT EXISTS idx_learning_policy_versions_model
    ON learning_policy_versions(source_model_version_id, created_at, policy_version_id);
CREATE INDEX IF NOT EXISTS idx_learning_policy_versions_dataset
    ON learning_policy_versions(source_dataset_version_id, created_at, policy_version_id);
CREATE INDEX IF NOT EXISTS idx_learning_policy_versions_fingerprint
    ON learning_policy_versions(fingerprint);

CREATE TRIGGER IF NOT EXISTS trg_learning_policy_versions_immutable_update
BEFORE UPDATE ON learning_policy_versions
BEGIN
    SELECT RAISE(ABORT, 'LearnedPolicyVersion records are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_policy_versions_immutable_delete
BEFORE DELETE ON learning_policy_versions
BEGIN
    SELECT RAISE(ABORT, 'LearnedPolicyVersion records are immutable');
END;

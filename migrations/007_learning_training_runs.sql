CREATE TABLE IF NOT EXISTS learning_training_runs (
    training_run_id TEXT PRIMARY KEY,
    training_identity_hash TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_version_label TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    train_count INTEGER NOT NULL CHECK (train_count >= 0),
    validation_count INTEGER NOT NULL CHECK (validation_count >= 0),
    oos_count INTEGER NOT NULL CHECK (oos_count >= 0),
    training_config_json TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    artifact_metadata_json TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    error_stage TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_training_runs_dataset
    ON learning_training_runs(dataset_version_id, created_at, training_run_id);
CREATE INDEX IF NOT EXISTS idx_learning_training_runs_status
    ON learning_training_runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_training_runs_identity
    ON learning_training_runs(training_identity_hash);

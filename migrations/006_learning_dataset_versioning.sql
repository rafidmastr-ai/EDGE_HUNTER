-- EDGE HUNTER L5: immutable learning dataset version manifests and row references.
CREATE TABLE IF NOT EXISTS learning_dataset_versions (
    dataset_version_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    dataset_policy_version TEXT NOT NULL,
    split_policy_version TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    train_count INTEGER NOT NULL CHECK (train_count >= 0),
    validation_count INTEGER NOT NULL CHECK (validation_count >= 0),
    oos_count INTEGER NOT NULL CHECK (oos_count >= 0),
    start_timestamp TEXT,
    end_timestamp TEXT,
    fingerprint TEXT NOT NULL UNIQUE,
    total_records INTEGER NOT NULL CHECK (total_records >= 0),
    included_records INTEGER NOT NULL CHECK (included_records >= 0),
    excluded_records INTEGER NOT NULL CHECK (excluded_records >= 0),
    statistics_json TEXT NOT NULL,
    filter_scope_json TEXT NOT NULL,
    split_policy_json TEXT NOT NULL,
    exclusion_counts_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    CHECK (row_count = included_records),
    CHECK (train_count + validation_count + oos_count = row_count),
    CHECK (total_records = included_records + excluded_records)
);

CREATE TABLE IF NOT EXISTS learning_dataset_rows (
    dataset_version_id TEXT NOT NULL REFERENCES learning_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    record_id TEXT NOT NULL REFERENCES learning_records(record_id) ON DELETE RESTRICT,
    split TEXT NOT NULL CHECK (split IN ('TRAIN', 'VALIDATION', 'OOS')),
    decision_timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_variant TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    feature_snapshot_id TEXT NOT NULL REFERENCES feature_snapshots(snapshot_id) ON DELETE RESTRICT,
    feature_snapshot_hash TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    label_version TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('HISTORICAL_BACKTEST', 'LIVE_TRADE')),
    row_fingerprint TEXT NOT NULL,
    row_metadata_json TEXT NOT NULL,
    PRIMARY KEY (dataset_version_id, record_id),
    UNIQUE (dataset_version_id, row_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_learning_dataset_versions_name
    ON learning_dataset_versions(dataset_name, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_dataset_rows_split
    ON learning_dataset_rows(dataset_version_id, split, decision_timestamp, record_id);
CREATE INDEX IF NOT EXISTS idx_learning_dataset_rows_record
    ON learning_dataset_rows(record_id);

CREATE TRIGGER IF NOT EXISTS trg_learning_dataset_versions_immutable_update
BEFORE UPDATE ON learning_dataset_versions
BEGIN
    SELECT RAISE(ABORT, 'learning dataset versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_dataset_versions_immutable_delete
BEFORE DELETE ON learning_dataset_versions
BEGIN
    SELECT RAISE(ABORT, 'learning dataset versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_dataset_rows_immutable_update
BEFORE UPDATE ON learning_dataset_rows
BEGIN
    SELECT RAISE(ABORT, 'learning dataset rows are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_learning_dataset_rows_immutable_delete
BEFORE DELETE ON learning_dataset_rows
BEGIN
    SELECT RAISE(ABORT, 'learning dataset rows are immutable');
END;

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    feature_engine_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    market_regime TEXT,
    mtf_context_json TEXT,
    snapshot_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    dataset_version TEXT,
    provenance_metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_identity
    ON feature_snapshots(symbol, timeframe, timestamp, feature_engine_version, feature_schema_version);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_created
    ON feature_snapshots(created_at);

ALTER TABLE learning_records
    ADD COLUMN feature_snapshot_id TEXT REFERENCES feature_snapshots(snapshot_id);

CREATE INDEX IF NOT EXISTS idx_learning_records_feature_snapshot
    ON learning_records(feature_snapshot_id);

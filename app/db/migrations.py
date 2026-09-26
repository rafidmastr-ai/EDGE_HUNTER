"""Database migration runner for EDGE HUNTER."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from app.db.database import Database


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATIONS = (
    Migration(
        1,
        "foundation",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """,
    ),
    Migration(
        2,
        "auth_subscriptions_admin",
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
            trial_started_at TEXT NOT NULL,
            trial_expires_at TEXT NOT NULL,
            device_hash TEXT,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );

        CREATE TABLE IF NOT EXISTS trial_devices (
            device_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            first_trial_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            starts_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked', 'expired')),
            source_code_id INTEGER,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            revoke_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS subscription_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_hash TEXT NOT NULL UNIQUE,
            code_hint TEXT NOT NULL,
            duration_months INTEGER NOT NULL CHECK (duration_months IN (1, 3)),
            status TEXT NOT NULL DEFAULT 'available' CHECK (status IN ('available', 'redeemed', 'revoked', 'expired')),
            assigned_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            expires_at TEXT,
            created_at TEXT NOT NULL,
            redeemed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS auth_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT NOT NULL UNIQUE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            scope TEXT NOT NULL CHECK (scope IN ('user', 'admin')),
            csrf_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);
        CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
        CREATE INDEX IF NOT EXISTS idx_subscription_codes_status ON subscription_codes(status);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
        """,
    ),
    Migration(
        3,
        "learning_foundation",
        """
        CREATE TABLE IF NOT EXISTS learning_records (
            record_id TEXT PRIMARY KEY,
            identity_hash TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL CHECK (source_type IN ('HISTORICAL_BACKTEST', 'LIVE_TRADE')),
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            decision_timestamp TEXT NOT NULL,
            entry_timestamp TEXT,
            direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
            entry_price REAL NOT NULL CHECK (entry_price > 0),
            target REAL NOT NULL CHECK (target > 0),
            stop_loss REAL NOT NULL CHECK (stop_loss > 0),
            risk_reward REAL NOT NULL CHECK (risk_reward > 0),
            strategy_name TEXT NOT NULL,
            strategy_variant TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            feature_set_version TEXT NOT NULL,
            parameters_snapshot_json TEXT NOT NULL,
            feature_snapshot_json TEXT NOT NULL,
            evidence_snapshot_json TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
            market_regime TEXT,
            outcome TEXT,
            exit_timestamp TEXT,
            exit_price REAL,
            exit_reason TEXT,
            duration_seconds INTEGER CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
            dataset_version TEXT,
            provenance_metadata_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING_OUTCOME'
                CHECK (status IN ('PENDING_OUTCOME', 'COMPLETED', 'INVALID', 'EXCLUDED')),
            created_at TEXT NOT NULL,
            CHECK (
                status <> 'COMPLETED'
                OR (
                    outcome IS NOT NULL
                    AND exit_timestamp IS NOT NULL
                    AND exit_price IS NOT NULL
                    AND exit_reason IS NOT NULL
                    AND duration_seconds IS NOT NULL
                )
            ),
            CHECK (
                status <> 'PENDING_OUTCOME'
                OR (
                    outcome IS NULL
                    AND exit_timestamp IS NULL
                    AND exit_price IS NULL
                    AND exit_reason IS NULL
                    AND duration_seconds IS NULL
                )
            )
        );

        CREATE INDEX IF NOT EXISTS idx_learning_records_status
            ON learning_records(status);
        CREATE INDEX IF NOT EXISTS idx_learning_records_source
            ON learning_records(source_type);
        CREATE INDEX IF NOT EXISTS idx_learning_records_decision
            ON learning_records(decision_timestamp);
        CREATE INDEX IF NOT EXISTS idx_learning_records_symbol_time
            ON learning_records(symbol, timeframe, decision_timestamp);
        """,
    ),
    Migration(
        4,
        "feature_snapshot_persistence",
        """
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
        """,
    ),
    Migration(
        5,
        "learning_outcome_label_version",
        """
        ALTER TABLE learning_records
            ADD COLUMN label_version TEXT;
        """,
    ),

    Migration(
        6,
        "learning_dataset_versioning",
        """
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
        """,
    ),
    Migration(
        7,
        "learning_training_runs",
        """
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
        """,
    ),
    Migration(
        8,
        "learning_model_registry",
        """
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
        """,
    ),
    Migration(
        9,
        "learning_l8_evaluations",
        (Path(__file__).resolve().parents[2] / "migrations" / "009_learning_l8_evaluations.sql").read_text(encoding="utf-8"),
    ),
    Migration(
        10,
        "learning_policy_versions",
        (Path(__file__).resolve().parents[2] / "migrations" / "010_learning_policy_versions.sql").read_text(encoding="utf-8"),
    ),
    Migration(
        11,
        "learning_continuous_learning",
        (Path(__file__).resolve().parents[2] / "migrations" / "011_learning_continuous_learning.sql").read_text(encoding="utf-8"),
    ),
)


class MigrationRunner:
    """Apply migrations exactly once in ascending version order."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def apply_all(self) -> None:
        self.database.connect()
        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.database.commit()

        applied = {
            row["version"]
            for row in self.database.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        }
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            self.database.connection.executescript(migration.sql)
            self.database.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
            self.database.commit()

"""Central configuration interface for EDGE HUNTER.

Secrets and admin credentials are intentionally loaded from secure runtime
configuration or created through the local admin bootstrap script.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Load local .env values without overwriting explicit environment variables."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is required to load the local .env file. "
            "Install project dependencies before starting EDGE HUNTER."
        ) from exc
    load_dotenv(env_file, override=False)


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class Settings:
    """Application settings used by foundation, web and authentication layers."""

    environment: str
    debug: bool
    database_path: Path
    api_host: str
    api_port: int
    live_provider_name: str
    live_provider_enabled: bool
    live_provider_url: str | None
    live_provider_api_key: str | None = field(repr=False)
    live_provider_timeout_seconds: float
    live_provider_max_retries: int
    live_provider_backoff_seconds: float
    live_provider_cache_ttl_seconds: float
    live_provider_max_bars: int
    live_provider_rate_limit_per_minute: int
    symbol_catalog_cache_ttl_seconds: int
    symbol_search_limit: int
    data_mode: str
    whatsapp_url: str | None
    admin_username: str | None
    session_ttl_hours: int
    password_iterations: int
    login_rate_limit: int
    rate_limit_window_seconds: int
    cookie_secure: bool
    allowed_hosts: tuple[str, ...]
    cors_origins: tuple[str, ...]
    public_rate_limit: int
    public_rate_limit_window_seconds: int
    max_request_bytes: int
    docs_enabled: bool
    trust_proxy_headers: bool
    log_dir: Path
    learning_enabled: bool
    learning_automatic_retraining_enabled: bool
    learning_automatic_promotion_enabled: bool
    learning_production_enabled: bool
    learning_schedule_interval_seconds: int
    learning_minimum_new_completed_records: int
    learning_minimum_dataset_size: int
    learning_cooldown_seconds: int
    learning_max_retries: int
    learning_minimum_monitoring_sample: int
    learning_drift_threshold: float
    learning_rollback_grace_period_seconds: int
    # Per-strategy machine learning (Classic/SMC/ICT). Models are produced only by
    # scripts/train_strategies.py and applied read-only by the web analysis.
    strategy_ml_enabled: bool = True
    strategy_ml_model_path: Path = PROJECT_ROOT / "data" / "models" / "strategy_learning.json"
    strategy_ml_record_live: bool = True
    strategy_ml_horizon_bars: int = 48


def load_settings() -> Settings:
    """Load non-secret settings from environment variables."""
    environment = os.getenv("EDGE_HUNTER_ENV", "development").strip().lower()
    if environment not in {"development", "test", "production"}:
        environment = "development"
    database_path = Path(
        os.getenv(
            "EDGE_HUNTER_DATABASE_PATH",
            str(PROJECT_ROOT / "data" / "edge_hunter.db"),
        )
    )

    raw_allowed_hosts = os.getenv("EDGE_HUNTER_ALLOWED_HOSTS")
    allowed_hosts = _env_csv("EDGE_HUNTER_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver")
    # Starlette TestClient uses the synthetic Host header ``testserver``.
    # Keep it available in development/test environments without weakening
    # production host validation.
    if environment != "production" and "testserver" not in allowed_hosts:
        allowed_hosts = (*allowed_hosts, "testserver")
    cors_origins = _env_csv("EDGE_HUNTER_CORS_ORIGINS")
    if "*" in cors_origins:
        raise ValueError("wildcard CORS origins are not allowed; use explicit origins")
    if environment == "production" and not raw_allowed_hosts:
        raise ValueError("EDGE_HUNTER_ALLOWED_HOSTS must be explicitly configured in production")
    # Live analysis is the default: OHLC comes from the live provider only.
    # "local" is an explicit offline mode (CSV files kept for backtest/learning).
    data_mode = os.getenv("EDGE_HUNTER_DATA_MODE", "live").strip().lower()
    if data_mode not in {"local", "live"}:
        raise ValueError("EDGE_HUNTER_DATA_MODE must be 'local' or 'live'")
    live_provider_name = os.getenv("EDGE_HUNTER_LIVE_PROVIDER", "none").strip().lower()
    if environment == "production" and data_mode == "live":
        if live_provider_name in {"", "none", "disabled"}:
            raise ValueError("a live provider must be configured when EDGE_HUNTER_DATA_MODE=live in production")
        if not _env_bool("EDGE_HUNTER_LIVE_PROVIDER_ENABLED", False):
            raise ValueError("EDGE_HUNTER_LIVE_PROVIDER_ENABLED must be true for production live mode")
        if not os.getenv("EDGE_HUNTER_LIVE_PROVIDER_API_KEY", "").strip():
            raise ValueError("EDGE_HUNTER_LIVE_PROVIDER_API_KEY must be configured for production live mode")

    return Settings(
        environment=environment,
        debug=_env_bool("EDGE_HUNTER_DEBUG", environment != "production"),
        database_path=database_path,
        api_host=os.getenv("EDGE_HUNTER_API_HOST", "127.0.0.1"),
        api_port=_env_int("EDGE_HUNTER_API_PORT", 8000, 1, 65535),
        live_provider_name=os.getenv("EDGE_HUNTER_LIVE_PROVIDER", "none").strip().lower(),
        live_provider_enabled=_env_bool("EDGE_HUNTER_LIVE_PROVIDER_ENABLED", False),
        live_provider_url=(
            os.getenv("EDGE_HUNTER_LIVE_PROVIDER_URL")
            if _env_bool("EDGE_HUNTER_LIVE_PROVIDER_ENABLED", False)
            or data_mode == "live"
            else None
        ),
        live_provider_api_key=os.getenv("EDGE_HUNTER_LIVE_PROVIDER_API_KEY"),
        live_provider_timeout_seconds=_env_float("EDGE_HUNTER_LIVE_PROVIDER_TIMEOUT_SECONDS", 10.0, 1.0, 60.0),
        live_provider_max_retries=_env_int("EDGE_HUNTER_LIVE_PROVIDER_MAX_RETRIES", 2, 0, 5),
        live_provider_backoff_seconds=_env_float("EDGE_HUNTER_LIVE_PROVIDER_BACKOFF_SECONDS", 0.5, 0.0, 10.0),
        live_provider_cache_ttl_seconds=_env_float("EDGE_HUNTER_LIVE_PROVIDER_CACHE_TTL_SECONDS", 15.0, 0.0, 300.0),
        live_provider_max_bars=_env_int("EDGE_HUNTER_LIVE_PROVIDER_MAX_BARS", 800, 50, 5000),
        live_provider_rate_limit_per_minute=_env_int("EDGE_HUNTER_LIVE_PROVIDER_RATE_LIMIT_PER_MINUTE", 8, 1, 500),
        symbol_catalog_cache_ttl_seconds=_env_int("EDGE_HUNTER_SYMBOL_CATALOG_CACHE_TTL_SECONDS", 86400, 300, 604800),
        symbol_search_limit=_env_int("EDGE_HUNTER_SYMBOL_SEARCH_LIMIT", 20, 5, 50),
        data_mode=data_mode,
        whatsapp_url=os.getenv("EDGE_HUNTER_WHATSAPP_URL"),
        admin_username=os.getenv("EDGE_HUNTER_ADMIN_USERNAME"),
        session_ttl_hours=_env_int("EDGE_HUNTER_SESSION_TTL_HOURS", 24, 1, 168),
        password_iterations=_env_int("EDGE_HUNTER_PASSWORD_ITERATIONS", 600_000, 100_000, 2_000_000),
        login_rate_limit=_env_int("EDGE_HUNTER_LOGIN_RATE_LIMIT", 5, 1, 30),
        rate_limit_window_seconds=_env_int("EDGE_HUNTER_RATE_LIMIT_WINDOW_SECONDS", 300, 30, 3600),
        cookie_secure=_env_bool("EDGE_HUNTER_COOKIE_SECURE", environment == "production"),
        allowed_hosts=allowed_hosts,
        cors_origins=cors_origins,
        public_rate_limit=_env_int("EDGE_HUNTER_PUBLIC_RATE_LIMIT", 120, 10, 10_000),
        public_rate_limit_window_seconds=_env_int("EDGE_HUNTER_PUBLIC_RATE_LIMIT_WINDOW_SECONDS", 60, 10, 3600),
        max_request_bytes=_env_int("EDGE_HUNTER_MAX_REQUEST_BYTES", 65_536, 1_024, 1_048_576),
        docs_enabled=_env_bool("EDGE_HUNTER_DOCS_ENABLED", environment != "production"),
        trust_proxy_headers=_env_bool("EDGE_HUNTER_TRUST_PROXY_HEADERS", False),
        log_dir=Path(os.getenv("EDGE_HUNTER_LOG_DIR", str(PROJECT_ROOT / "logs"))),
        learning_enabled=_env_bool("EDGE_HUNTER_LEARNING_ENABLED", False),
        learning_automatic_retraining_enabled=_env_bool("EDGE_HUNTER_LEARNING_AUTOMATIC_RETRAINING_ENABLED", False),
        learning_automatic_promotion_enabled=_env_bool("EDGE_HUNTER_LEARNING_AUTOMATIC_PROMOTION_ENABLED", False),
        learning_production_enabled=_env_bool("EDGE_HUNTER_LEARNING_PRODUCTION_ENABLED", False),
        learning_schedule_interval_seconds=_env_int("EDGE_HUNTER_LEARNING_SCHEDULE_INTERVAL_SECONDS", 3600, 60, 86400),
        learning_minimum_new_completed_records=_env_int("EDGE_HUNTER_LEARNING_MINIMUM_NEW_COMPLETED_RECORDS", 1, 0, 1_000_000),
        learning_minimum_dataset_size=_env_int("EDGE_HUNTER_LEARNING_MINIMUM_DATASET_SIZE", 1, 0, 1_000_000),
        learning_cooldown_seconds=_env_int("EDGE_HUNTER_LEARNING_COOLDOWN_SECONDS", 3600, 0, 31_536_000),
        learning_max_retries=_env_int("EDGE_HUNTER_LEARNING_MAX_RETRIES", 2, 0, 10),
        learning_minimum_monitoring_sample=_env_int("EDGE_HUNTER_LEARNING_MINIMUM_MONITORING_SAMPLE", 10, 0, 1_000_000),
        learning_drift_threshold=_env_float("EDGE_HUNTER_LEARNING_DRIFT_THRESHOLD", 0.25, 0.0, 10.0),
        learning_rollback_grace_period_seconds=_env_int("EDGE_HUNTER_LEARNING_ROLLBACK_GRACE_PERIOD_SECONDS", 3600, 0, 31_536_000),
        strategy_ml_enabled=_env_bool("EDGE_HUNTER_STRATEGY_ML_ENABLED", True),
        strategy_ml_model_path=Path(
            os.getenv(
                "EDGE_HUNTER_STRATEGY_ML_MODEL_PATH",
                str(PROJECT_ROOT / "data" / "models" / "strategy_learning.json"),
            )
        ),
        strategy_ml_record_live=_env_bool("EDGE_HUNTER_STRATEGY_ML_RECORD_LIVE", True),
        strategy_ml_horizon_bars=_env_int("EDGE_HUNTER_STRATEGY_ML_HORIZON_BARS", 48, 5, 500),
    )

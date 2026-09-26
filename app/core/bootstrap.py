"""Application bootstrap for Phase 01."""

from __future__ import annotations

from dataclasses import dataclass

from config.config_hunter import Settings, load_settings
from app.db.database import Database
from app.db.migrations import MigrationRunner
from app.providers.csv_provider import CsvHistoricalDataProvider
from app.providers.factory import build_live_provider
from app.providers.live_provider import LiveMarketDataProvider


@dataclass
class Application:
    """Composition root for the foundation components."""

    settings: Settings
    database: Database
    historical_provider: CsvHistoricalDataProvider
    live_provider: LiveMarketDataProvider

    def start(self) -> None:
        """Initialize infrastructure required by the application."""
        self.database.connect()
        MigrationRunner(self.database).apply_all()


def create_application() -> Application:
    """Build the application dependency graph."""
    settings = load_settings()
    database = Database(settings.database_path)
    return Application(
        settings=settings,
        database=database,
        historical_provider=CsvHistoricalDataProvider(),
        live_provider=build_live_provider(settings),
    )

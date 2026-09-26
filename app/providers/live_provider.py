"""Provider-neutral live market-data interfaces and safe null implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.domain.market import OHLCBar
from app.providers.models import LiveProviderError, LiveProviderHealth


class LiveMarketDataProvider(Protocol):
    """Canonical live-provider boundary consumed by the application."""

    name: str

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCBar]:
        """Return canonical OHLC bars for a requested range."""
        ...

    def health(self) -> LiveProviderHealth:
        """Return safe operational status without exposing credentials."""
        ...

    def get_symbol_catalog(self) -> list[dict[str, str]]:
        """Return provider-supported Forex/Crypto/Metal reference symbols."""
        ...


class UnavailableLiveMarketDataProvider:
    """Explicit safe state used until a real provider is configured."""

    name = "none"

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCBar]:
        raise LiveProviderError(
            "live market data provider is not configured",
            code="live_provider_unconfigured",
            status_code=503,
        )

    def health(self) -> LiveProviderHealth:
        return LiveProviderHealth(
            provider=self.name,
            configured=False,
            status="unconfigured",
        )

    def get_symbol_catalog(self) -> list[dict[str, str]]:
        raise LiveProviderError(
            "live market data provider is not configured",
            code="live_provider_unconfigured",
            status_code=503,
        )

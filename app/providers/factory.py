"""Factory for selecting the configured live market-data adapter."""

from __future__ import annotations

from typing import Any

from app.providers.live_provider import LiveMarketDataProvider, UnavailableLiveMarketDataProvider
from app.providers.twelvedata import TwelveDataLiveProvider


def build_live_provider(settings: Any) -> LiveMarketDataProvider:
    """Build the configured live provider without exposing provider secrets."""
    provider_name = (settings.live_provider_name or "none").strip().lower()
    if not settings.live_provider_enabled or provider_name in {"", "none", "disabled"}:
        return UnavailableLiveMarketDataProvider()
    if provider_name == "twelvedata":
        return TwelveDataLiveProvider(
            api_key=settings.live_provider_api_key,
            base_url=settings.live_provider_url or "https://api.twelvedata.com",
            timeout_seconds=settings.live_provider_timeout_seconds,
            max_retries=settings.live_provider_max_retries,
            backoff_seconds=settings.live_provider_backoff_seconds,
            cache_ttl_seconds=settings.live_provider_cache_ttl_seconds,
            max_bars=settings.live_provider_max_bars,
            rate_limit_per_minute=settings.live_provider_rate_limit_per_minute,
            rate_limit_wait_seconds=settings.live_provider_rate_limit_wait_seconds,
        )
    raise ValueError(f"unsupported live provider: {provider_name}")

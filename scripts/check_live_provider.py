"""Explicit opt-in smoke test for the configured live provider.

This script performs a real provider request only when live mode is explicitly
enabled and a provider API key is present. It never prints the API key.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from app.providers.factory import build_live_provider
from app.providers.models import LiveProviderError
from config.config_hunter import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER live-provider smoke test")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--bars", type=int, default=50)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.live_provider_enabled:
        print("LIVE PROVIDER CHECK BLOCKED: EDGE_HUNTER_LIVE_PROVIDER_ENABLED is false")
        return 2
    if not settings.live_provider_api_key:
        print("LIVE PROVIDER CHECK BLOCKED: API key is not configured")
        return 2

    provider = build_live_provider(settings)
    end = datetime.now(timezone.utc)
    minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}[args.timeframe.upper()]
    start = end - timedelta(minutes=minutes * max(10, min(args.bars, 5000)))

    try:
        bars = provider.get_ohlc(args.symbol.upper(), args.timeframe.upper(), start, end)
    except (LiveProviderError, ValueError, KeyError) as exc:
        print(f"LIVE PROVIDER CHECK FAILED: {exc.code if isinstance(exc, LiveProviderError) else 'configuration'}")
        return 1

    health = provider.health().to_dict()
    print(f"LIVE PROVIDER CHECK PASS: {provider.name}")
    print(f"Symbol: {args.symbol.upper()}")
    print(f"Timeframe: {args.timeframe.upper()}")
    print(f"Bars received: {len(bars)}")
    print(f"First timestamp: {bars[0].timestamp.isoformat()}")
    print(f"Last timestamp: {bars[-1].timestamp.isoformat()}")
    print(f"Health: {health['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
